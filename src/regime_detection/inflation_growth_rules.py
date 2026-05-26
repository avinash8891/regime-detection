from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from regime_detection.config import InflationGrowthRulesConfig
from regime_detection.inflation_growth import (
    InflationGrowthLabel,
    InflationGrowthRuleInputs,
)


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def _credit_is_calm(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    if inputs.credit_funding_active_label == "credit_calm":
        return True
    if inputs.credit_funding_active_label is None and getattr(
        config, "allow_credit_independent_fallback", False
    ):
        return True
    return False


@dataclass(frozen=True)
class GoldilocksLimbEvidence:
    credit_is_calm: bool
    drift_ok: bool
    drift_margin: float | None
    slope_ok: bool
    slope_margin: float | None
    benign_ok: bool
    benign_margin: float | None
    limb_count: int

    def as_evidence(self) -> dict[str, object]:
        return asdict(self)


def goldilocks_limb_evidence(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> GoldilocksLimbEvidence:
    credit_is_calm = _credit_is_calm(inputs, config)

    drift_margin: float | None = None
    drift_ok = False
    if not _any_nan(inputs.cpi_6m_change_pct, inputs.cpi_6m_change_pct_lag_21):
        drift = abs(inputs.cpi_6m_change_pct - inputs.cpi_6m_change_pct_lag_21)
        drift_margin = config.cpi_drift_threshold - drift
        drift_ok = drift_margin >= 0.0

    slope_margin: float | None = None
    slope_ok = False
    if not np.isnan(inputs.cpi_6m_change_pct_slope_21d):
        slope_margin = 0.0 - inputs.cpi_6m_change_pct_slope_21d
        slope_ok = slope_margin >= 0.0

    benign_margin: float | None = None
    benign_ok = False
    ceiling = getattr(config, "cpi_goldilocks_benign_ceiling", None)
    if ceiling is not None and not np.isnan(inputs.cpi_6m_change_pct):
        benign_margin = ceiling - inputs.cpi_6m_change_pct
        benign_ok = benign_margin > 0.0

    limb_count = sum((drift_ok, slope_ok, benign_ok))
    return GoldilocksLimbEvidence(
        credit_is_calm=credit_is_calm,
        drift_ok=drift_ok,
        drift_margin=drift_margin,
        slope_ok=slope_ok,
        slope_margin=slope_margin,
        benign_ok=benign_ok,
        benign_margin=benign_margin,
        limb_count=limb_count,
    )


def evaluate_goldilocks(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3047-3053 — `goldilocks` rule."""
    limb_evidence = goldilocks_limb_evidence(inputs, config)
    if not limb_evidence.credit_is_calm:
        return False
    if _any_nan(
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
    ):
        return False
    if limb_evidence.limb_count == 0:
        return False
    return bool(
        inputs.pmi_manufacturing > config.pmi_goldilocks_threshold
        and inputs.spy_21d_return > 0.0
    )


def evaluate_inflation_shock(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3055-3062 — `inflation_shock` three-limb OR rule.

    Limb 1: surprise z-score (single-signal).
    Limb 2: commodity + yield + equity composite.
    Limb 3: rapid 3m CPI acceleration with rising yields — catches
    inflation onset before the 6m window absorbs it.
    """
    if not _any_nan(inputs.inflation_surprise_zscore) and (
        inputs.inflation_surprise_zscore > config.inflation_surprise_zscore_threshold
    ):
        return True

    if not _any_nan(
        inputs.commodity_return_63d,
        inputs.treasury_10y_yield_slope_21d,
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
    ) and bool(
        inputs.commodity_return_63d > config.commodity_return_threshold
        and inputs.treasury_10y_yield_slope_21d > 0.0
        and inputs.spy_21d_return < 0.0
        and inputs.tlt_21d_return < 0.0
    ):
        return True

    if not _any_nan(
        inputs.cpi_3m_change_pct,
        inputs.treasury_10y_yield_slope_21d,
    ) and bool(
        inputs.cpi_3m_change_pct > config.cpi_3m_acceleration_threshold
        and inputs.treasury_10y_yield_slope_21d > 0.0
    ):
        return True

    return False


def evaluate_disinflation(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3064-3067 — `disinflation` rule."""
    if _any_nan(inputs.cpi_6m_change_pct_slope_21d, inputs.pmi_manufacturing):
        return False
    if inputs.pmi_manufacturing <= config.pmi_disinflation_threshold:
        return False
    if inputs.cpi_6m_change_pct_slope_21d >= 0.0:
        return False
    if getattr(config, "disinflation_yield_independent", False):
        return True
    if np.isnan(inputs.treasury_10y_yield_slope_21d):
        return False
    return bool(inputs.treasury_10y_yield_slope_21d < 0.0)


def _credit_is_stressed(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    if inputs.credit_funding_active_label in {"spread_widening", "credit_stress"}:
        return True
    if inputs.credit_funding_active_label is None and getattr(
        config, "allow_credit_independent_fallback", False
    ):
        return True
    return False


def evaluate_recession_scare(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3069-3077 — `recession_scare` rule."""
    if _any_nan(
        inputs.treasury_10y_yield_slope_21d,
        inputs.cyclical_defensive_slope_21d,
        inputs.spy_21d_return,
    ):
        return False
    if inputs.credit_funding_active_label in {"spread_widening", "credit_stress"}:
        threshold = getattr(
            config,
            "spy_recession_credit_confirmed_threshold",
            config.spy_recession_threshold,
        )
        return bool(
            inputs.treasury_10y_yield_slope_21d < 0.0
            and inputs.cyclical_defensive_slope_21d < 0.0
            and inputs.spy_21d_return < threshold
        )
    if inputs.credit_funding_active_label is None and getattr(
        config, "allow_credit_independent_fallback", False
    ):
        threshold = getattr(
            config,
            "spy_recession_credit_independent_threshold",
            config.spy_recession_threshold,
        )
        return bool(
            inputs.treasury_10y_yield_slope_21d < 0.0
            and inputs.cyclical_defensive_slope_21d < 0.0
            and inputs.spy_21d_return < threshold
        )
    return False


def evaluate_recovery_growth(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3079-3082 — `recovery_growth` rule."""
    if not _credit_is_calm(inputs, config):
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


def evaluate_risk_off_mild(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Mild risk-off: credit stressed, equity declining, AND at least one
    growth-deterioration signal from the real economy.

    Requires credit = spread_widening/credit_stress, spy declining but
    not crashing, PLUS at least one of:
      - cyclical/defensive rotation negative (risk-off sector flow)
      - treasury yields falling (flight to safety)
      - PMI below expansion threshold (manufacturing contracting)
    """
    if inputs.credit_funding_active_label not in {"spread_widening", "credit_stress"}:
        return False
    if _any_nan(inputs.spy_21d_return):
        return False
    threshold = getattr(
        config,
        "spy_recession_credit_confirmed_threshold",
        config.spy_recession_threshold,
    )
    if not (inputs.spy_21d_return < 0.0 and inputs.spy_21d_return >= threshold):
        return False
    growth_deterioration = False
    if not np.isnan(inputs.cyclical_defensive_slope_21d):
        growth_deterioration = (
            growth_deterioration or inputs.cyclical_defensive_slope_21d < 0.0
        )
    if not np.isnan(inputs.treasury_10y_yield_slope_21d):
        growth_deterioration = (
            growth_deterioration or inputs.treasury_10y_yield_slope_21d < 0.0
        )
    if not np.isnan(inputs.pmi_manufacturing):
        growth_deterioration = (
            growth_deterioration
            or inputs.pmi_manufacturing < config.pmi_goldilocks_threshold
        )
    return growth_deterioration


def _credit_not_crisis(inputs: InflationGrowthRuleInputs) -> bool:
    """True when credit is absent, calm, recovering, or mildly widening —
    any state short of outright stress / squeeze / deleveraging."""
    return inputs.credit_funding_active_label not in {
        "credit_stress",
        "funding_squeeze",
        "deleveraging",
    }


def evaluate_reflation(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Rising CPI + economic expansion + equities positive + credit not in crisis.

    Captures the "normal growth with mild inflation pressure" regime that
    falls between goldilocks (requires credit_calm) and recession_scare
    (requires equity decline).
    """
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d > 0.0
        and inputs.pmi_manufacturing > config.pmi_goldilocks_threshold
        and inputs.spy_21d_return > 0.0
        and _credit_not_crisis(inputs)
    )


def evaluate_stagflation_lite(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Rising CPI + contracting manufacturing (PMI ≤ 50).

    Captures the early-warning macro regime where inflation persists while
    the real economy weakens. Distinct from recession_scare (requires equity
    decline + credit stress) and from reflation (requires PMI > 50).
    """
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.pmi_manufacturing,
    ):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d > 0.0
        and inputs.pmi_manufacturing <= config.pmi_goldilocks_threshold
    )


def evaluate_earnings_expansion(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 3100-3102 — strict positive aggregate forward-EPS revision."""
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
    """v2 §2B lines 3104-3106 — strict negative aggregate forward-EPS revision."""
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
    if evaluate_risk_off_mild(inputs, config):
        return "risk_off_mild"
    if evaluate_disinflation(inputs, config):
        return "disinflation"
    if evaluate_goldilocks(inputs, config):
        return "goldilocks"
    if evaluate_recovery_growth(inputs, config):
        return "recovery_growth"
    if evaluate_reflation(inputs, config):
        return "reflation"
    if evaluate_stagflation_lite(inputs, config):
        return "stagflation_lite"
    if evaluate_earnings_contraction(inputs, config):
        return "earnings_contraction"
    if evaluate_earnings_expansion(inputs, config):
        return "earnings_expansion"
    return "macro_mixed"
