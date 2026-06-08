"""v2 §2B Inflation/Growth axis — classify layer.

Labels, risk rank, required-input keys, per-day rule input materialisation,
predicates, and the precedence walker. The features layer lives in
``inflation_growth.py`` and ships ``InflationGrowthFeatures`` +
``compute_inflation_growth_features``; this module consumes that dataclass
to build per-day rule inputs and evaluate predicates.

Spec references (docs/regime_engine_v2_spec.md):
    §2B lines 2965-2975   Labels
    §2B line  2980        Precedence:
        inflation_shock > recession_scare > risk_off_mild > disinflation >
        goldilocks > recovery_growth > reflation > stagflation_lite >
        earnings_contraction > earnings_expansion > unknown
    §2B lines 2983-3107   Rules
    §2B lines 3109-3124   Risk Rank

Required-input keys (CPI_KEY, PMI_KEY, ...) live here as the public
vocabulary axis_builders / feature_store consume to thread the right
macro_series + cross_asset_closes into the features function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import (
    any_nan as _any_nan,
    scalar_at as _scalar_at,
    scalar_at_lag as _scalar_at_lag,
)
from regime_detection._series_alignment import aligned_float_values
from regime_detection.config import InflationGrowthRulesConfig
from regime_detection.inflation_growth import InflationGrowthFeatures

# ---------------------------------------------------------------------------
# Spec labels (V2 §2B spec lines 2965-2975) + risk rank (V2 §2B spec lines 3109-3124).
# ---------------------------------------------------------------------------

InflationGrowthLabel = Literal[
    "goldilocks",
    "inflation_shock",
    "disinflation",
    "recession_scare",
    "risk_off_mild",
    "recovery_growth",
    "recovery_growth_unconfirmed",
    "reflation",
    "late_cycle_inflation_stress",
    "stagflation_lite",
    "contractionary_disinflation",
    "macro_neutral",
    "earnings_expansion",
    "earnings_contraction",
    "unknown",
]


INFLATION_GROWTH_RISK_RANK: dict[InflationGrowthLabel, int] = {
    "goldilocks": 0,
    "recovery_growth": 0,
    "earnings_expansion": 0,
    "recovery_growth_unconfirmed": 1,
    "reflation": 1,
    "macro_neutral": 1,
    "unknown": 1,
    "disinflation": 1,
    "contractionary_disinflation": 2,
    "late_cycle_inflation_stress": 2,
    "stagflation_lite": 2,
    "risk_off_mild": 2,
    "earnings_contraction": 2,
    "recession_scare": 3,
    "inflation_shock": 3,
}


# ---------------------------------------------------------------------------
# Required input keys. Pinned here as single source of truth.
# ---------------------------------------------------------------------------

CPI_KEY = "cpi_all_items"
PMI_KEY = "pmi_manufacturing"
DGS10_KEY = "10y_yield"
DBC_KEY = "DBC"
TLT_KEY = "TLT"
XLY_KEY = "XLY"
XLI_KEY = "XLI"
XLP_KEY = "XLP"
XLU_KEY = "XLU"
CPI_NOWCAST_KEY = "cpi_nowcast"
AGG_FORWARD_EPS_REVISION_KEY = "aggregate_forward_eps_revision"

REQUIRED_CROSS_ASSET_KEYS: tuple[str, ...] = (
    DBC_KEY,
    TLT_KEY,
    XLY_KEY,
    XLI_KEY,
    XLP_KEY,
    XLU_KEY,
)
REQUIRED_MACRO_KEYS: tuple[str, ...] = (CPI_KEY, PMI_KEY, DGS10_KEY)


# ---------------------------------------------------------------------------
# Per-day scalar rule inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationGrowthRuleInputs:
    """Per-day scalars consumed by the §2B rule predicates.

    ``credit_funding_active_label`` carries the cross-axis dependency from
    §2C; ``None`` signals the §2C axis is unbuilt (cross-axis short-circuit
    per V2 §2B "Cross-Axis Short-Circuit" subsection ~spec line 3159).
    """

    cpi_3m_change_pct: float
    cpi_6m_change_pct: float
    cpi_6m_change_pct_lag_21: float
    cpi_6m_change_pct_slope_21d: float
    inflation_surprise_zscore: float
    aggregate_forward_eps_revision_direction_4w: float
    pmi_manufacturing: float
    pmi_manufacturing_slope_21d: float
    commodity_return_63d: float
    treasury_10y_yield_slope_21d: float
    cyclical_defensive_slope_21d: float
    spy_21d_return: float
    tlt_21d_return: float
    credit_funding_active_label: str | None


def build_rule_inputs_for_date(
    *,
    features: InflationGrowthFeatures,
    dt: pd.Timestamp,
    config: InflationGrowthRulesConfig,
    credit_funding_active_label: str | None,
) -> InflationGrowthRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``."""
    return InflationGrowthRuleInputs(
        cpi_3m_change_pct=_scalar_at(features.cpi_3m_change_pct, dt),
        cpi_6m_change_pct=_scalar_at(features.cpi_6m_change_pct, dt),
        cpi_6m_change_pct_lag_21=_scalar_at_lag(
            features.cpi_6m_change_pct, dt, config.cpi_slope_lookback_sessions
        ),
        cpi_6m_change_pct_slope_21d=_scalar_at(
            features.cpi_6m_change_pct_slope_21d, dt
        ),
        inflation_surprise_zscore=_scalar_at(features.inflation_surprise_zscore, dt),
        aggregate_forward_eps_revision_direction_4w=_scalar_at(
            features.aggregate_forward_eps_revision_direction_4w, dt
        ),
        pmi_manufacturing=_scalar_at(features.pmi_manufacturing, dt),
        pmi_manufacturing_slope_21d=_scalar_at(
            features.pmi_manufacturing_slope_21d, dt
        ),
        commodity_return_63d=_scalar_at(features.commodity_return_63d, dt),
        treasury_10y_yield_slope_21d=_scalar_at(
            features.treasury_10y_yield_slope_21d, dt
        ),
        cyclical_defensive_slope_21d=_scalar_at(
            features.cyclical_defensive_slope_21d, dt
        ),
        spy_21d_return=_scalar_at(features.spy_21d_return, dt),
        tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
        credit_funding_active_label=credit_funding_active_label,
    )


def build_rule_inputs_by_date(
    *,
    features: InflationGrowthFeatures,
    config: InflationGrowthRulesConfig,
    credit_funding_active_labels_by_date: dict[pd.Timestamp, str | None] | None,
) -> dict[pd.Timestamp, InflationGrowthRuleInputs]:
    index = features.cpi_6m_change_pct.index
    cpi_lag_21 = features.cpi_6m_change_pct.shift(config.cpi_slope_lookback_sessions)
    cpi_3m_values = aligned_float_values(features.cpi_3m_change_pct, index)
    cpi_6m_values = aligned_float_values(features.cpi_6m_change_pct, index)
    cpi_lag_values = aligned_float_values(cpi_lag_21, index)
    cpi_slope_values = aligned_float_values(features.cpi_6m_change_pct_slope_21d, index)
    inflation_surprise_values = aligned_float_values(
        features.inflation_surprise_zscore, index
    )
    eps_revision_values = aligned_float_values(
        features.aggregate_forward_eps_revision_direction_4w, index
    )
    pmi_values = aligned_float_values(features.pmi_manufacturing, index)
    pmi_slope_values = aligned_float_values(features.pmi_manufacturing_slope_21d, index)
    commodity_return_values = aligned_float_values(features.commodity_return_63d, index)
    treasury_slope_values = aligned_float_values(
        features.treasury_10y_yield_slope_21d, index
    )
    cyclical_defensive_slope_values = aligned_float_values(
        features.cyclical_defensive_slope_21d, index
    )
    spy_return_values = aligned_float_values(features.spy_21d_return, index)
    tlt_return_values = aligned_float_values(features.tlt_21d_return, index)

    outputs: dict[pd.Timestamp, InflationGrowthRuleInputs] = {}
    for pos, dt in enumerate(index):
        credit_funding_active_label = None
        if credit_funding_active_labels_by_date is not None:
            credit_funding_active_label = credit_funding_active_labels_by_date.get(dt)
        outputs[dt] = InflationGrowthRuleInputs(
            cpi_3m_change_pct=float(cpi_3m_values[pos]),
            cpi_6m_change_pct=float(cpi_6m_values[pos]),
            cpi_6m_change_pct_lag_21=float(cpi_lag_values[pos]),
            cpi_6m_change_pct_slope_21d=float(cpi_slope_values[pos]),
            inflation_surprise_zscore=float(inflation_surprise_values[pos]),
            aggregate_forward_eps_revision_direction_4w=float(eps_revision_values[pos]),
            pmi_manufacturing=float(pmi_values[pos]),
            pmi_manufacturing_slope_21d=float(pmi_slope_values[pos]),
            commodity_return_63d=float(commodity_return_values[pos]),
            treasury_10y_yield_slope_21d=float(treasury_slope_values[pos]),
            cyclical_defensive_slope_21d=float(cyclical_defensive_slope_values[pos]),
            spy_21d_return=float(spy_return_values[pos]),
            tlt_21d_return=float(tlt_return_values[pos]),
            credit_funding_active_label=credit_funding_active_label,
        )
    return outputs


# ---------------------------------------------------------------------------
# Rule predicates (§2B lines 2983-3107).
# ---------------------------------------------------------------------------


def _credit_is_calm(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    if inputs.credit_funding_active_label == "credit_calm":
        return True
    if (
        inputs.credit_funding_active_label is None
        and config.allow_credit_independent_fallback
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
    if not np.isnan(inputs.cpi_6m_change_pct):
        benign_margin = config.cpi_goldilocks_benign_ceiling - inputs.cpi_6m_change_pct
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
    if config.disinflation_yield_independent:
        return True
    if np.isnan(inputs.treasury_10y_yield_slope_21d):
        return False
    return bool(inputs.treasury_10y_yield_slope_21d < 0.0)


def evaluate_contractionary_disinflation(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Demand contraction with non-rising inflation pressure."""
    if _any_nan(inputs.cpi_6m_change_pct_slope_21d, inputs.pmi_manufacturing):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d <= 0.0
        and inputs.pmi_manufacturing <= config.pmi_disinflation_threshold
    )


def _credit_is_stressed(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    if inputs.credit_funding_active_label in {"spread_widening", "credit_stress"}:
        return True
    if (
        inputs.credit_funding_active_label is None
        and config.allow_credit_independent_fallback
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
        threshold = config.spy_recession_credit_confirmed_threshold
        return bool(
            inputs.treasury_10y_yield_slope_21d < 0.0
            and inputs.cyclical_defensive_slope_21d < 0.0
            and inputs.spy_21d_return < threshold
        )
    if (
        inputs.credit_funding_active_label is None
        and config.allow_credit_independent_fallback
    ):
        threshold = config.spy_recession_credit_independent_threshold
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


def evaluate_recovery_growth_unconfirmed(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Improving growth where credit is not calm enough for confirmed recovery."""
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.pmi_manufacturing_slope_21d,
        inputs.pmi_manufacturing,
        inputs.cyclical_defensive_slope_21d,
    ):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d <= 0.0
        and inputs.pmi_manufacturing_slope_21d > 0.0
        and inputs.pmi_manufacturing > config.pmi_recovery_threshold
        and inputs.cyclical_defensive_slope_21d > 0.0
        and inputs.credit_funding_active_label
        in {"credit_divergence", "credit_recovery", "spread_widening"}
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
    threshold = config.spy_recession_credit_confirmed_threshold
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


def evaluate_late_cycle_inflation_stress(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """Expansion with non-declining inflation and market/credit stress."""
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
    ):
        return False
    credit_stress = inputs.credit_funding_active_label in {
        "credit_stress",
        "funding_squeeze",
        "deleveraging",
    }
    return bool(
        inputs.cpi_6m_change_pct_slope_21d >= 0.0
        and inputs.pmi_manufacturing > config.pmi_goldilocks_threshold
        and (inputs.spy_21d_return <= 0.0 or credit_stress)
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


def evaluate_macro_neutral(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,  # noqa: ARG001
) -> bool:
    """Finite core macro inputs with no directional rule impulse."""
    return not _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
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
    if evaluate_goldilocks(inputs, config):
        return "goldilocks"
    if evaluate_recovery_growth(inputs, config):
        return "recovery_growth"
    if evaluate_recovery_growth_unconfirmed(inputs, config):
        return "recovery_growth_unconfirmed"
    if evaluate_disinflation(inputs, config):
        return "disinflation"
    if evaluate_late_cycle_inflation_stress(inputs, config):
        return "late_cycle_inflation_stress"
    if evaluate_reflation(inputs, config):
        return "reflation"
    if evaluate_stagflation_lite(inputs, config):
        return "stagflation_lite"
    if evaluate_earnings_contraction(inputs, config):
        return "earnings_contraction"
    if evaluate_earnings_expansion(inputs, config):
        return "earnings_expansion"
    if evaluate_contractionary_disinflation(inputs, config):
        return "contractionary_disinflation"
    if evaluate_macro_neutral(inputs, config):
        return "macro_neutral"
    return "unknown"
