"""v2 §3.4–§3.5 Network Fragility rule engine + precedence.

Pure scalar rule layer over the features produced by
``regime_detection.network_fragility.compute_features``.

Spec references (docs/regime_engine_v2_spec.md):
    §3.3 Labels        (line 600–608)
    §3.4 Precedence    (line 610–613)
    §3.5 Rules         (line 615–657)
    §3.6 Risk Rank     (line 659–669)

The six rules are evaluated in §3.4 precedence order; the first match wins.
If none match, the label falls through to ``unknown`` (consistent with §3.3
and the risk-rank table in §3.6).

Cross-axis inputs:
    - ``breadth_label`` from V1 ``BreadthLabel`` (regime_detection.breadth_state)
    - ``volatility_label`` from V1 ``VolatilityLabel`` (regime_detection.volatility_state)
    - ``credit_funding_label`` from V2 §2C credit/funding axis.
      When ``credit_funding_label is None`` the ``systemic_stress`` rule
      short-circuits to ``False`` and precedence falls through to
      ``correlation_to_one``.

Slope detection for ``rising_fragility``:
    OLS slope of the trailing 21d window of the feature vs a unit trading-day
    index. A strictly positive slope signals "rising". The 21d window is part
    of the rule input materialization (see ``build_rule_inputs_for_date``).

Effective-rank stability for ``diversified_normal``:
    std(effective_rank over trailing 21d) / mean(...) < 0.05. The 0.05
    threshold is configurable via ``NetworkFragilityRulesConfig``.

Module invariant:
    All numeric thresholds are config-driven via ``NetworkFragilityRulesConfig``;
    spec-fixed window lengths (21d slope window, 21d stability window, 21d
    drawdown window) and the strict-positive slope comparator are module
    constants per v2 §3.5.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from regime_detection.breadth_state import BreadthLabel
from regime_detection.config import NetworkFragilityRulesConfig
from regime_detection.network_fragility import NetworkFragilityFeatures
from regime_detection.volatility_state import VolatilityLabel


# v2 §3.3 labels.
NetworkFragilityLabel = Literal[
    "diversified_normal",
    "stock_picker_dispersion",
    "rising_fragility",
    "correlation_concentration",
    "correlation_to_one",
    "systemic_stress",
    "unknown",
]


# v2 §3.4: systemic_stress > correlation_to_one > correlation_concentration
#          > rising_fragility > stock_picker_dispersion > diversified_normal > unknown
RULE_PRECEDENCE: tuple[NetworkFragilityLabel, ...] = (
    "systemic_stress",
    "correlation_to_one",
    "correlation_concentration",
    "rising_fragility",
    "stock_picker_dispersion",
    "diversified_normal",
)


# v2 §3.6 lines 661–668: risk rank for asymmetric hysteresis. Verbatim from
# the spec (NOT a tunable). `systemic_stress` shares rank 3 with
# `correlation_to_one`, and `rising_fragility` shares rank 2 with
# `correlation_concentration`. `unknown` is mid-rank (2) so it neither
# fast-tracks escalation past correlation_to_one nor strands the engine
# in a low-risk label across NaN gaps.
NETWORK_FRAGILITY_RISK_RANK: dict[NetworkFragilityLabel, int] = {
    "diversified_normal": 0,
    "stock_picker_dispersion": 1,
    "rising_fragility": 2,
    "correlation_concentration": 2,
    "correlation_to_one": 3,
    "systemic_stress": 3,
    "unknown": 2,
}


# v2 §2C credit/funding labels (formal enum lives in credit_funding.py).
# Re-declared here as a local Literal alias to avoid a circular import (the
# §2C classifier consumes nothing from this module).
CreditFundingLabel = Literal[
    "credit_calm",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


# Window lengths fixed by spec text in §3.5 ("rising over 21d", "21d std",
# "drawdown_21d"). These are spec constants, not tunables.
_SPEC_SLOPE_WINDOW_DAYS = 21
_SPEC_STABILITY_WINDOW_DAYS = 21
_SPEC_DRAWDOWN_WINDOW_DAYS = 21


@dataclass(frozen=True)
class NetworkFragilityRuleInputs:
    """Per-day scalar inputs the §3.5 rules consume.

    Materialized from a NetworkFragilityFeatures series + cross-axis label
    series + (V1 volatility) vix_percentile_252d at a single date by
    ``build_rule_inputs_for_date``. Keeping rules scalar makes them easy to
    test in isolation against §3.5 thresholds.
    """

    # §3.2 raw + percentile features at session t.
    avg_pairwise_corr_percentile_504d: float
    largest_eigenvalue_share_percentile_504d: float
    effective_rank_percentile_504d: float
    dispersion_ratio_percentile_252d: float

    # §3.2 absorption ratio — top-3 eigenvalue concentration.
    absorption_ratio_top3: float

    # §3.5 rising_fragility slopes (positive => rising) over trailing 21d.
    avg_pairwise_corr_slope_21d: float
    largest_eigenvalue_share_slope_21d: float

    # §3.5 diversified_normal stability: std/mean of effective_rank over 21d.
    effective_rank_stability_21d: float

    # §3.5 correlation_to_one / systemic_stress cross-axis scalars.
    realized_vol_percentile_252d: float
    drawdown_21d: float
    vix_percentile_252d: float


def _trailing_slope(series: pd.Series, dt: pd.Timestamp, window: int) -> float:
    """OLS slope of ``series`` vs a unit trading-day index over the trailing
    ``window`` sessions ending at ``dt`` (inclusive). NaN if window not full."""
    sub = series.loc[:dt].tail(window)
    if len(sub) < window:
        return float("nan")
    y = sub.to_numpy(dtype=float)
    if np.isnan(y).any():
        return float("nan")
    x = np.arange(window, dtype=float)
    # polyfit deg=1 returns [slope, intercept]. Suppress RankWarning emitted
    # on flat / near-constant inputs — it is CI noise; the returned slope is
    # still numerically correct (effectively zero) for our predicate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _trailing_stability(series: pd.Series, dt: pd.Timestamp, window: int) -> float:
    """std / mean of ``series`` over the trailing ``window`` sessions
    ending at ``dt`` (inclusive). NaN on insufficient data or zero mean."""
    sub = series.loc[:dt].tail(window)
    if len(sub) < window:
        return float("nan")
    arr = sub.to_numpy(dtype=float)
    if np.isnan(arr).any():
        return float("nan")
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    return float(arr.std(ddof=0) / mean)


def _trailing_drawdown(spy_close: pd.Series, dt: pd.Timestamp, window: int) -> float:
    """drawdown_21d per v2 §3.5: (P_t / max(P over trailing 21d)) - 1.

    Negative means below the 21d high. NaN on insufficient data."""
    sub = spy_close.loc[:dt].tail(window)
    if len(sub) < window:
        return float("nan")
    arr = sub.to_numpy(dtype=float)
    if np.isnan(arr).any():
        return float("nan")
    peak = arr.max()
    if peak <= 0:
        return float("nan")
    return float(arr[-1] / peak - 1.0)


def build_rule_inputs_for_date(
    *,
    features: NetworkFragilityFeatures,
    dt: pd.Timestamp,
    spy_close: pd.Series,
    realized_vol_percentile_252d: pd.Series,
    vix_percentile_252d: pd.Series,
) -> NetworkFragilityRuleInputs:
    """Materialize per-day scalar inputs for the §3.5 rules.

    All windows are fixed by §3.5 text (21d). Series-to-scalar reduction
    lives here so the rule functions stay pure scalar predicates.
    """
    return NetworkFragilityRuleInputs(
        avg_pairwise_corr_percentile_504d=float(
            features.avg_pairwise_corr_percentile_504d.loc[dt]
        ),
        largest_eigenvalue_share_percentile_504d=float(
            features.largest_eigenvalue_share_percentile_504d.loc[dt]
        ),
        effective_rank_percentile_504d=float(
            features.effective_rank_percentile_504d.loc[dt]
        ),
        dispersion_ratio_percentile_252d=float(
            features.dispersion_ratio_percentile_252d.loc[dt]
        ),
        absorption_ratio_top3=float(
            features.absorption_ratio_top3.loc[dt]
        ),
        avg_pairwise_corr_slope_21d=_trailing_slope(
            features.avg_pairwise_corr_63d, dt, _SPEC_SLOPE_WINDOW_DAYS
        ),
        largest_eigenvalue_share_slope_21d=_trailing_slope(
            features.largest_eigenvalue_share, dt, _SPEC_SLOPE_WINDOW_DAYS
        ),
        effective_rank_stability_21d=_trailing_stability(
            features.effective_rank, dt, _SPEC_STABILITY_WINDOW_DAYS
        ),
        realized_vol_percentile_252d=float(realized_vol_percentile_252d.loc[dt]),
        drawdown_21d=_trailing_drawdown(spy_close, dt, _SPEC_DRAWDOWN_WINDOW_DAYS),
        vix_percentile_252d=float(vix_percentile_252d.loc[dt]),
    )


def _rolling_ols_slope_series(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=series.index)

    windows = sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)
    if valid.any():
        x = np.arange(window, dtype=float)
        x_sum = float(x.sum())
        x_sq_sum = float(np.square(x).sum())
        denom = window * x_sq_sum - x_sum * x_sum
        valid_windows = windows[valid]
        y_sum = valid_windows.sum(axis=1)
        xy_sum = valid_windows @ x
        out[window - 1 :][valid] = (
            window * xy_sum - x_sum * y_sum
        ) / denom
    return pd.Series(out, index=series.index)


def _rolling_stability_series(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=series.index)

    windows = sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)
    if valid.any():
        valid_windows = windows[valid]
        means = valid_windows.mean(axis=1)
        stabilities = np.full(len(valid_windows), np.nan, dtype=float)
        nonzero = means != 0.0
        if nonzero.any():
            stabilities[nonzero] = (
                valid_windows[nonzero].std(axis=1, ddof=0) / means[nonzero]
            )
        out[window - 1 :][valid] = stabilities
    return pd.Series(out, index=series.index)


def _rolling_drawdown_series(spy_close: pd.Series, window: int) -> pd.Series:
    peak = spy_close.rolling(window=window, min_periods=window).max()
    drawdown = spy_close / peak - 1.0
    drawdown = drawdown.where(peak > 0)
    return drawdown.astype(float)


def build_rule_inputs_by_date(
    *,
    features: NetworkFragilityFeatures,
    spy_close: pd.Series,
    realized_vol_percentile_252d: pd.Series,
    vix_percentile_252d: pd.Series,
) -> dict[pd.Timestamp, NetworkFragilityRuleInputs]:
    index = features.avg_pairwise_corr_63d.index
    avg_corr_slope = _rolling_ols_slope_series(
        features.avg_pairwise_corr_63d, _SPEC_SLOPE_WINDOW_DAYS
    )
    largest_eig_slope = _rolling_ols_slope_series(
        features.largest_eigenvalue_share, _SPEC_SLOPE_WINDOW_DAYS
    )
    eff_rank_stability = _rolling_stability_series(
        features.effective_rank, _SPEC_STABILITY_WINDOW_DAYS
    )
    drawdown = _rolling_drawdown_series(spy_close.reindex(index), _SPEC_DRAWDOWN_WINDOW_DAYS)
    realized_vol = realized_vol_percentile_252d.reindex(index)
    vix_pct = vix_percentile_252d.reindex(index)

    outputs: dict[pd.Timestamp, NetworkFragilityRuleInputs] = {}
    for dt in index:
        outputs[dt] = NetworkFragilityRuleInputs(
            avg_pairwise_corr_percentile_504d=float(
                features.avg_pairwise_corr_percentile_504d.loc[dt]
            ),
            largest_eigenvalue_share_percentile_504d=float(
                features.largest_eigenvalue_share_percentile_504d.loc[dt]
            ),
            effective_rank_percentile_504d=float(
                features.effective_rank_percentile_504d.loc[dt]
            ),
            dispersion_ratio_percentile_252d=float(
                features.dispersion_ratio_percentile_252d.loc[dt]
            ),
            absorption_ratio_top3=float(
                features.absorption_ratio_top3.loc[dt]
            ),
            avg_pairwise_corr_slope_21d=float(avg_corr_slope.loc[dt]),
            largest_eigenvalue_share_slope_21d=float(largest_eig_slope.loc[dt]),
            effective_rank_stability_21d=float(eff_rank_stability.loc[dt]),
            realized_vol_percentile_252d=float(realized_vol.loc[dt]),
            drawdown_21d=float(drawdown.loc[dt]),
            vix_percentile_252d=float(vix_pct.loc[dt]),
        )
    return outputs


# -- Rule predicates (v2 §3.5) -------------------------------------------------


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def evaluate_diversified_normal(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
) -> bool:
    """v2 §3.5 lines 617–621.

    `0.25 <= avg_pairwise_corr_percentile_504d <= 0.75
     AND effective_rank stable (21d std < 5% of mean)`
    """
    if _any_nan(
        inputs.avg_pairwise_corr_percentile_504d,
        inputs.effective_rank_stability_21d,
    ):
        return False
    in_band = (
        config.diversified_normal_percentile_lo
        <= inputs.avg_pairwise_corr_percentile_504d
        <= config.diversified_normal_percentile_hi
    )
    stable = (
        inputs.effective_rank_stability_21d < config.effective_rank_stability_threshold
    )
    return bool(in_band and stable)


def evaluate_stock_picker_dispersion(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
    volatility_label: VolatilityLabel,
) -> bool:
    """v2 §3.5 lines 623–628.

    `avg_pairwise_corr_percentile_504d < 0.30
     AND dispersion_ratio percentile_252d > 0.70
     AND volatility_state.active_label != crisis_vol`
    """
    if _any_nan(
        inputs.avg_pairwise_corr_percentile_504d,
        inputs.dispersion_ratio_percentile_252d,
    ):
        return False
    return bool(
        inputs.avg_pairwise_corr_percentile_504d < config.stock_picker_percentile_max
        and inputs.dispersion_ratio_percentile_252d
        > config.stock_picker_dispersion_percentile_min
        and volatility_label != "crisis_vol"
    )


def evaluate_rising_fragility(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,  # noqa: ARG001 (kept for uniform signature)
    breadth_label: BreadthLabel,
) -> bool:
    """v2 §3.5 lines 630–635.

    `avg_pairwise_corr rising over 21d (positive slope)
     AND largest_eigenvalue_share rising over 21d
     AND breadth_state.active_label in [weak_breadth, narrowing_breadth, divergent_fragile]`

    Note: v2 §3.5 line 634 references `narrowing_breadth`. The
    ``BreadthLabel`` enum includes `narrowing_breadth` alongside
    `weak_breadth` and `divergent_fragile`, so the accepted_breadth
    set matches the spec text verbatim.
    """
    if _any_nan(
        inputs.avg_pairwise_corr_slope_21d,
        inputs.largest_eigenvalue_share_slope_21d,
    ):
        return False
    accepted_breadth: set[BreadthLabel] = {
        "weak_breadth",
        "narrowing_breadth",
        "divergent_fragile",
    }
    return bool(
        inputs.avg_pairwise_corr_slope_21d > 0.0
        and inputs.largest_eigenvalue_share_slope_21d > 0.0
        and breadth_label in accepted_breadth
    )


def evaluate_correlation_concentration(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
) -> bool:
    """v2 §3.5 lines 637–642 + absorption_ratio_top3 extension.

    `avg_pairwise_corr_percentile_504d > 0.75
     OR largest_eigenvalue_share_percentile_504d > 0.75
     OR effective_rank_percentile_504d < 0.25
     OR absorption_ratio_top3 > 0.90`
    """
    cond_corr = (
        not np.isnan(inputs.avg_pairwise_corr_percentile_504d)
        and inputs.avg_pairwise_corr_percentile_504d
        > config.concentration_corr_percentile_min
    )
    cond_eig = (
        not np.isnan(inputs.largest_eigenvalue_share_percentile_504d)
        and inputs.largest_eigenvalue_share_percentile_504d
        > config.concentration_largest_eig_percentile_min
    )
    cond_rank = (
        not np.isnan(inputs.effective_rank_percentile_504d)
        and inputs.effective_rank_percentile_504d
        < config.concentration_effective_rank_percentile_max
    )
    cond_absorption = (
        not np.isnan(inputs.absorption_ratio_top3)
        and inputs.absorption_ratio_top3
        > config.concentration_absorption_ratio_min
    )
    return bool(cond_corr or cond_eig or cond_rank or cond_absorption)


def evaluate_correlation_to_one(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
) -> bool:
    """v2 §3.5 lines 644–649.

    `avg_pairwise_corr_percentile_504d > 0.90
     AND realized_vol_percentile_252d > 0.80
     AND drawdown_21d < 0`
    """
    if _any_nan(
        inputs.avg_pairwise_corr_percentile_504d,
        inputs.realized_vol_percentile_252d,
        inputs.drawdown_21d,
    ):
        return False
    return bool(
        inputs.avg_pairwise_corr_percentile_504d
        > config.corr_to_one_corr_percentile_min
        and inputs.realized_vol_percentile_252d
        > config.corr_to_one_realized_vol_percentile_min
        and inputs.drawdown_21d < config.corr_to_one_drawdown_max
    )


def evaluate_systemic_stress(
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
    breadth_label: BreadthLabel,
    credit_funding_label: CreditFundingLabel | None,
) -> bool:
    """v2 §3.5 lines 651–657.

    `correlation_to_one
     AND credit_funding.active_label in [credit_stress, deleveraging]
     AND VIX_percentile_252d > 0.80
     AND breadth_state.active_label in [weak_breadth, narrowing_breadth]`

    Short-circuit: when ``credit_funding_label is None`` (credit/funding
    seam not lit) this rule is False. Precedence then falls through to
    ``correlation_to_one``.
    """
    if credit_funding_label is None:
        return False
    # Explicit NaN guard over ALL fields this rule relies on — mirrors the
    # pattern in the other §3.5 rules and decouples correctness from
    # `evaluate_correlation_to_one`'s transitive NaN handling. `vix_percentile_252d`
    # is additionally guarded here because it is unique to systemic_stress.
    if _any_nan(
        inputs.avg_pairwise_corr_percentile_504d,
        inputs.realized_vol_percentile_252d,
        inputs.drawdown_21d,
        inputs.vix_percentile_252d,
    ):
        return False
    if not evaluate_correlation_to_one(inputs, config):
        return False
    accepted_credit: set[CreditFundingLabel] = {"credit_stress", "deleveraging"}
    # v2 §3.5 line 656: accepted breadth set matches spec verbatim.
    accepted_breadth: set[BreadthLabel] = {"weak_breadth", "narrowing_breadth"}
    return bool(
        credit_funding_label in accepted_credit
        and inputs.vix_percentile_252d > config.systemic_stress_vix_percentile_min
        and breadth_label in accepted_breadth
    )


def evaluate_rules(
    *,
    inputs: NetworkFragilityRuleInputs,
    config: NetworkFragilityRulesConfig,
    breadth_label: BreadthLabel,
    volatility_label: VolatilityLabel,
    credit_funding_label: CreditFundingLabel | None = None,
) -> NetworkFragilityLabel:
    """Walk the v2 §3.4 precedence and return the first matching label.

    Falls through to ``unknown`` (v2 §3.3) when no rule fires.
    """
    for label in RULE_PRECEDENCE:
        if label == "systemic_stress":
            if evaluate_systemic_stress(
                inputs, config, breadth_label, credit_funding_label
            ):
                return "systemic_stress"
        elif label == "correlation_to_one":
            if evaluate_correlation_to_one(inputs, config):
                return "correlation_to_one"
        elif label == "correlation_concentration":
            if evaluate_correlation_concentration(inputs, config):
                return "correlation_concentration"
        elif label == "rising_fragility":
            if evaluate_rising_fragility(inputs, config, breadth_label):
                return "rising_fragility"
        elif label == "stock_picker_dispersion":
            if evaluate_stock_picker_dispersion(inputs, config, volatility_label):
                return "stock_picker_dispersion"
        elif label == "diversified_normal":
            if evaluate_diversified_normal(inputs, config):
                return "diversified_normal"
    return "unknown"
