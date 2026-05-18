"""TDD tests for v2 §3.4–§3.5 Network Fragility rule engine + precedence.

Per ~/.claude/CLAUDE.md and AGENTS rule A:
- Realistic v2 label values (BreadthLabel/VolatilityLabel/CreditFundingLabel
  enums imported from real modules).
- Real spec percentiles (0.25, 0.30, 0.45, 0.55, 0.70, 0.75, 0.80, 0.90).
- Integration test invokes the rule engine end-to-end over a real
  NetworkFragilityFeatures series.
"""
from __future__ import annotations


import pytest

from regime_detection.config import NetworkFragilityRulesConfig, load_default_regime_config
from regime_detection.network_fragility_rules import (
    NetworkFragilityRuleInputs,
    evaluate_correlation_concentration,
    evaluate_correlation_to_one,
    evaluate_diversified_normal,
    evaluate_rising_fragility,
    evaluate_stock_picker_dispersion,
    evaluate_systemic_stress,
)


# ---------- Helpers -----------------------------------------------------------


def _default_rules_config() -> NetworkFragilityRulesConfig:
    return load_default_regime_config().network_fragility.rules


def _inputs(
    *,
    avg_corr_pct: float = 0.50,
    largest_eig_pct: float = 0.50,
    eff_rank_pct: float = 0.50,
    dispersion_pct: float = 0.50,
    avg_corr_slope: float = 0.0,
    largest_eig_slope: float = 0.0,
    eff_rank_stability: float = 0.02,
    realized_vol_pct: float = 0.50,
    drawdown_21d: float = 0.0,
    vix_pct: float = 0.50,
) -> NetworkFragilityRuleInputs:
    """Construct rule inputs with sane mid-band defaults; override only
    the dimensions a test is exercising."""
    return NetworkFragilityRuleInputs(
        avg_pairwise_corr_percentile_504d=avg_corr_pct,
        largest_eigenvalue_share_percentile_504d=largest_eig_pct,
        effective_rank_percentile_504d=eff_rank_pct,
        dispersion_ratio_percentile_252d=dispersion_pct,
        avg_pairwise_corr_slope_21d=avg_corr_slope,
        largest_eigenvalue_share_slope_21d=largest_eig_slope,
        effective_rank_stability_21d=eff_rank_stability,
        realized_vol_percentile_252d=realized_vol_pct,
        drawdown_21d=drawdown_21d,
        vix_percentile_252d=vix_pct,
    )


# ---------- Config wiring -----------------------------------------------------


def test_default_yaml_loads_rules_block_with_spec_thresholds():
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    rules = cfg.network_fragility.rules
    # v2 §3.5 verbatim thresholds.
    assert rules.diversified_normal_percentile_lo == 0.25
    assert rules.diversified_normal_percentile_hi == 0.75
    assert rules.effective_rank_stability_threshold == 0.05
    assert rules.stock_picker_percentile_max == 0.30
    assert rules.stock_picker_dispersion_percentile_min == 0.70
    assert rules.concentration_corr_percentile_min == 0.75
    assert rules.concentration_largest_eig_percentile_min == 0.75
    assert rules.concentration_effective_rank_percentile_max == 0.25
    assert rules.corr_to_one_corr_percentile_min == 0.90
    assert rules.corr_to_one_realized_vol_percentile_min == 0.80
    assert rules.corr_to_one_drawdown_max == 0.0
    assert rules.systemic_stress_vix_percentile_min == 0.80


def test_rules_config_rejects_unknown_keys():
    from pydantic import ValidationError

    cfg = load_default_regime_config()
    base = cfg.network_fragility.rules.model_dump()
    base["unexpected_threshold"] = 0.5
    with pytest.raises(ValidationError):
        NetworkFragilityRulesConfig.model_validate(base)


# ---------- Individual rule predicates ---------------------------------------


def test_diversified_normal_fires_inside_25_75_band_and_stable_effective_rank():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.50, eff_rank_stability=0.01)
    assert evaluate_diversified_normal(inputs, cfg) is True


def test_diversified_normal_excludes_percentile_above_75():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.80, eff_rank_stability=0.01)
    assert evaluate_diversified_normal(inputs, cfg) is False


def test_diversified_normal_excludes_unstable_effective_rank():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.50, eff_rank_stability=0.10)
    assert evaluate_diversified_normal(inputs, cfg) is False


def test_stock_picker_dispersion_fires_on_low_corr_high_dispersion():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.20, dispersion_pct=0.85)
    assert (
        evaluate_stock_picker_dispersion(inputs, cfg, volatility_label="normal_vol")
        is True
    )


def test_stock_picker_dispersion_blocked_by_crisis_volatility():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.20, dispersion_pct=0.85)
    assert (
        evaluate_stock_picker_dispersion(inputs, cfg, volatility_label="crisis_vol")
        is False
    )


def test_stock_picker_dispersion_blocked_when_corr_above_threshold():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.35, dispersion_pct=0.85)
    assert (
        evaluate_stock_picker_dispersion(inputs, cfg, volatility_label="normal_vol")
        is False
    )


def test_rising_fragility_fires_on_positive_slopes_and_weak_breadth():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_slope=0.001, largest_eig_slope=0.0005)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="weak_breadth") is True
    )


def test_rising_fragility_fires_on_divergent_fragile_breadth():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_slope=0.001, largest_eig_slope=0.0005)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="divergent_fragile")
        is True
    )


def test_rising_fragility_fires_on_narrowing_breadth():
    """v2 §3.5 line 634 names `narrowing_breadth` in the accepted breadth set.
    Slice 2.8c widened the `BreadthLabel` enum to include `narrowing_breadth`;
    the Log #3 TODO follow-up adds it to the rule's accepted_breadth set so
    the spec text and the rule predicate now agree."""
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_slope=0.001, largest_eig_slope=0.0005)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="narrowing_breadth")
        is True
    )


def test_rising_fragility_blocked_by_healthy_breadth():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_slope=0.001, largest_eig_slope=0.0005)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="healthy_breadth")
        is False
    )


def test_rising_fragility_blocked_by_flat_slope():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_slope=0.0, largest_eig_slope=0.0005)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="weak_breadth") is False
    )


def test_correlation_concentration_fires_when_corr_percentile_above_75():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.80, largest_eig_pct=0.40, eff_rank_pct=0.50)
    assert evaluate_correlation_concentration(inputs, cfg) is True


def test_correlation_concentration_fires_when_largest_eig_above_75():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.50, largest_eig_pct=0.85, eff_rank_pct=0.50)
    assert evaluate_correlation_concentration(inputs, cfg) is True


def test_correlation_concentration_fires_when_effective_rank_below_25():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.50, largest_eig_pct=0.40, eff_rank_pct=0.10)
    assert evaluate_correlation_concentration(inputs, cfg) is True


def test_correlation_concentration_does_not_fire_in_mid_band():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.55, largest_eig_pct=0.55, eff_rank_pct=0.55)
    assert evaluate_correlation_concentration(inputs, cfg) is False


def test_correlation_to_one_requires_all_three_conditions():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.95, realized_vol_pct=0.85, drawdown_21d=-0.04)
    assert evaluate_correlation_to_one(inputs, cfg) is True


def test_correlation_to_one_blocked_by_positive_drawdown():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.95, realized_vol_pct=0.85, drawdown_21d=0.01)
    assert evaluate_correlation_to_one(inputs, cfg) is False


def test_correlation_to_one_blocked_by_low_realized_vol_percentile():
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.95, realized_vol_pct=0.70, drawdown_21d=-0.05)
    assert evaluate_correlation_to_one(inputs, cfg) is False


def test_systemic_stress_short_circuits_to_false_when_credit_funding_label_absent():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    assert (
        evaluate_systemic_stress(
            inputs,
            cfg,
            breadth_label="weak_breadth",
            credit_funding_label=None,
        )
        is False
    )


def test_systemic_stress_fires_when_all_conditions_met():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    assert (
        evaluate_systemic_stress(
            inputs,
            cfg,
            breadth_label="weak_breadth",
            credit_funding_label="credit_stress",
        )
        is True
    )


def test_systemic_stress_fires_on_narrowing_breadth():
    """v2 §3.5 line 656 names `narrowing_breadth` in the accepted breadth set
    alongside `weak_breadth`. Log #3 TODO follow-up: the rule's
    accepted_breadth set now includes `narrowing_breadth` so the predicate
    matches the spec text verbatim."""
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    assert (
        evaluate_systemic_stress(
            inputs,
            cfg,
            breadth_label="narrowing_breadth",
            credit_funding_label="credit_stress",
        )
        is True
    )


def test_systemic_stress_blocked_by_neutral_credit_funding():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    assert (
        evaluate_systemic_stress(
            inputs,
            cfg,
            breadth_label="weak_breadth",
            credit_funding_label="neutral_funding",
        )
        is False
    )


# ---------- Boundary / NaN tests (strict-inequality thresholds) --------------


def test_stock_picker_excludes_corr_pct_exactly_at_threshold():
    """v2 §3.5 line 625: `avg_pairwise_corr_percentile_504d < 0.30` is
    strict; equality must NOT fire stock_picker."""
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.30, dispersion_pct=0.85)
    assert (
        evaluate_stock_picker_dispersion(inputs, cfg, volatility_label="normal_vol")
        is False
    )


def test_correlation_concentration_excludes_corr_pct_exactly_at_threshold():
    """v2 §3.5 line 639: `> 0.75` is strict; equality at 0.75 must NOT fire."""
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.75, largest_eig_pct=0.50, eff_rank_pct=0.50)
    assert evaluate_correlation_concentration(inputs, cfg) is False


def test_correlation_to_one_excludes_corr_pct_exactly_at_threshold():
    """v2 §3.5 line 646: `> 0.90` is strict; equality at 0.90 must NOT fire."""
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.90, realized_vol_pct=0.85, drawdown_21d=-0.04)
    assert evaluate_correlation_to_one(inputs, cfg) is False


def test_correlation_to_one_excludes_realized_vol_pct_exactly_at_threshold():
    """v2 §3.5 line 647: `> 0.80` is strict; equality at 0.80 must NOT fire."""
    cfg = _default_rules_config()
    inputs = _inputs(avg_corr_pct=0.95, realized_vol_pct=0.80, drawdown_21d=-0.04)
    assert evaluate_correlation_to_one(inputs, cfg) is False


def test_systemic_stress_blocked_by_nan_vix_percentile():
    """N-2 / I-2: explicit NaN guard on `vix_percentile_252d`. With ALL other
    fields valid AND credit_funding='credit_stress' (so the credit short-circuit
    does not fire), a NaN VIX percentile must produce False rather than
    silently relying on `evaluate_correlation_to_one`'s NaN handling."""
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=float("nan"),
    )
    assert (
        evaluate_systemic_stress(
            inputs,
            cfg,
            breadth_label="weak_breadth",
            credit_funding_label="credit_stress",
        )
        is False
    )


# ---------- Precedence orchestrator ------------------------------------------
