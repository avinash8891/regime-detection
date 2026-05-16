"""TDD tests for v2 §3.4–§3.5 Network Fragility rule engine + precedence.

Per ~/.claude/CLAUDE.md and AGENTS rule A:
- Realistic v2 label values (BreadthLabel/VolatilityLabel/CreditFundingLabel
  enums imported from real modules).
- Real spec percentiles (0.25, 0.30, 0.45, 0.55, 0.70, 0.75, 0.80, 0.90).
- Integration test invokes the rule engine end-to-end over a real
  NetworkFragilityFeatures series.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import NetworkFragilityRulesConfig, load_default_regime_config
from regime_detection.network_fragility import (
    NetworkFragilityFeatures,
    compute_features,
)
from regime_detection.network_fragility_rules import (
    RULE_PRECEDENCE,
    NetworkFragilityRuleInputs,
    build_rule_inputs_by_date,
    build_rule_inputs_for_date,
    evaluate_correlation_concentration,
    evaluate_correlation_to_one,
    evaluate_diversified_normal,
    evaluate_rising_fragility,
    evaluate_rules,
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


def test_precedence_ordering_is_spec_3_4():
    # v2 §3.4 line 612.
    assert RULE_PRECEDENCE == (
        "systemic_stress",
        "correlation_to_one",
        "correlation_concentration",
        "rising_fragility",
        "stock_picker_dispersion",
        "diversified_normal",
    )


def test_systemic_stress_beats_correlation_to_one():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="weak_breadth",
        volatility_label="high_vol",
        credit_funding_label="credit_stress",
    )
    assert label == "systemic_stress"


def test_systemic_stress_falls_through_to_correlation_to_one_when_credit_funding_absent():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        vix_pct=0.90,
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="weak_breadth",
        volatility_label="high_vol",
        credit_funding_label=None,
    )
    assert label == "correlation_to_one"


def test_correlation_to_one_beats_correlation_concentration():
    cfg = _default_rules_config()
    # avg_corr 0.95 satisfies both (concentration > 0.75 AND corr_to_one > 0.90).
    inputs = _inputs(
        avg_corr_pct=0.95,
        realized_vol_pct=0.85,
        drawdown_21d=-0.04,
        largest_eig_pct=0.85,
        eff_rank_pct=0.20,
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="neutral_breadth",
        volatility_label="high_vol",
    )
    assert label == "correlation_to_one"


def test_correlation_concentration_beats_rising_fragility():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.80,                 # triggers concentration
        avg_corr_slope=0.001,              # would trigger rising_fragility
        largest_eig_slope=0.001,
        realized_vol_pct=0.50,             # NOT triggering corr_to_one (need 0.80)
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="weak_breadth",
        volatility_label="normal_vol",
    )
    assert label == "correlation_concentration"


def test_rising_fragility_beats_stock_picker_dispersion():
    cfg = _default_rules_config()
    # Low corr_pct (0.20) would trigger stock_picker; but positive slope +
    # weak_breadth triggers rising_fragility first.
    inputs = _inputs(
        avg_corr_pct=0.20,
        dispersion_pct=0.85,
        avg_corr_slope=0.001,
        largest_eig_slope=0.001,
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="weak_breadth",
        volatility_label="normal_vol",
    )
    assert label == "rising_fragility"


def test_stock_picker_dispersion_when_only_it_matches():
    cfg = _default_rules_config()
    # corr_pct = 0.20 is BELOW the diversified_normal band [0.25, 0.75], so
    # diversified_normal does NOT fire here; only stock_picker matches.
    # This test exercises the single-rule match path (not precedence).
    inputs = _inputs(
        avg_corr_pct=0.20,
        dispersion_pct=0.85,
        eff_rank_stability=0.01,
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "stock_picker_dispersion"


def test_stock_picker_precedence_when_both_match():
    """Genuine precedence collision: avg_corr_pct=0.26 sits INSIDE the
    diversified_normal band [0.25, 0.75] AND below the stock_picker threshold
    (< 0.30); dispersion_pct=0.85 satisfies stock_picker's high-dispersion
    leg; eff_rank_stability is tight enough that diversified_normal would
    also fire on its own. Per §3.4 precedence, stock_picker_dispersion wins."""
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.26,
        dispersion_pct=0.85,
        eff_rank_stability=0.01,
    )
    # Sanity: both individual predicates fire on these inputs.
    assert evaluate_diversified_normal(inputs, cfg) is True
    assert (
        evaluate_stock_picker_dispersion(inputs, cfg, volatility_label="normal_vol")
        is True
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "stock_picker_dispersion"


def test_diversified_normal_when_only_it_matches():
    cfg = _default_rules_config()
    inputs = _inputs(
        avg_corr_pct=0.50,
        eff_rank_stability=0.01,
        dispersion_pct=0.40,    # under stock_picker dispersion threshold
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "diversified_normal"


def test_unknown_when_no_rule_matches():
    cfg = _default_rules_config()
    # Pick a feature profile that satisfies no rule.
    inputs = _inputs(
        avg_corr_pct=0.20,            # below diversified_normal band
        dispersion_pct=0.40,          # below stock_picker dispersion
        avg_corr_slope=-0.001,        # negative slope
        largest_eig_slope=-0.001,
        largest_eig_pct=0.40,
        eff_rank_pct=0.50,            # not below 0.25
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "unknown"


# ---------- build_rule_inputs_for_date ---------------------------------------


def test_build_rule_inputs_for_date_computes_positive_slope_for_rising_corr():
    """Synthetic rising avg_pairwise_corr series → positive slope at last
    session; flat largest_eigenvalue_share → zero slope."""
    index = pd.bdate_range(end="2024-12-31", periods=50)
    avg_corr = pd.Series(np.linspace(0.30, 0.60, 50), index=index)
    flat_eig = pd.Series(0.50, index=index)
    eff_rank = pd.Series(np.linspace(4.0, 4.0, 50), index=index)  # zero variance
    pct = pd.Series(0.50, index=index)
    spy_close = pd.Series(np.linspace(400.0, 420.0, 50), index=index)
    vix_pct = pd.Series(0.50, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=avg_corr,
        avg_pairwise_corr_percentile_504d=pct,
        largest_eigenvalue_share=flat_eig,
        largest_eigenvalue_share_percentile_504d=pct,
        effective_rank=eff_rank,
        effective_rank_percentile_504d=pct,
        absorption_ratio_top3=pct,
        dispersion_ratio=pct,
        dispersion_ratio_percentile_252d=pct,
    )
    realized_vol_pct = pd.Series(0.50, index=index)

    inputs = build_rule_inputs_for_date(
        features=features,
        dt=index[-1],
        spy_close=spy_close,
        realized_vol_percentile_252d=realized_vol_pct,
        vix_percentile_252d=vix_pct,
    )
    assert inputs.avg_pairwise_corr_slope_21d > 0
    assert inputs.largest_eigenvalue_share_slope_21d == pytest.approx(0.0, abs=1e-15)
    assert math.isnan(inputs.effective_rank_stability_21d) or (
        inputs.effective_rank_stability_21d == pytest.approx(0.0, abs=1e-12)
    )


def test_build_rule_inputs_for_date_drawdown_is_negative_when_below_peak():
    index = pd.bdate_range(end="2024-12-31", periods=50)
    # Peak at session -10, then drop 5%.
    spy = np.concatenate([np.linspace(400.0, 420.0, 40), np.linspace(420.0, 399.0, 10)])
    spy_close = pd.Series(spy, index=index)
    flat = pd.Series(0.50, index=index)
    flat_eff_rank = pd.Series(4.0, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=flat,
        avg_pairwise_corr_percentile_504d=flat,
        largest_eigenvalue_share=flat,
        largest_eigenvalue_share_percentile_504d=flat,
        effective_rank=flat_eff_rank,
        effective_rank_percentile_504d=flat,
        absorption_ratio_top3=flat,
        dispersion_ratio=flat,
        dispersion_ratio_percentile_252d=flat,
    )
    inputs = build_rule_inputs_for_date(
        features=features,
        dt=index[-1],
        spy_close=spy_close,
        realized_vol_percentile_252d=flat,
        vix_percentile_252d=flat,
    )
    # Last price 399 < peak 420 within trailing 21d → drawdown < 0.
    assert inputs.drawdown_21d < 0


def test_build_rule_inputs_by_date_matches_single_day_builder():
    index = pd.bdate_range(end="2024-12-31", periods=80)
    avg_corr = pd.Series(np.linspace(0.30, 0.60, 80), index=index)
    eig = pd.Series(np.linspace(0.40, 0.55, 80), index=index)
    eff_rank = pd.Series(np.linspace(4.0, 5.0, 80), index=index)
    pct = pd.Series(np.linspace(0.20, 0.90, 80), index=index)
    spy_close = pd.Series(np.linspace(400.0, 430.0, 80), index=index)
    realized_vol_pct = pd.Series(np.linspace(0.10, 0.80, 80), index=index)
    vix_pct = pd.Series(np.linspace(0.15, 0.85, 80), index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=avg_corr,
        avg_pairwise_corr_percentile_504d=pct,
        largest_eigenvalue_share=eig,
        largest_eigenvalue_share_percentile_504d=pct,
        effective_rank=eff_rank,
        effective_rank_percentile_504d=pct,
        absorption_ratio_top3=pct,
        dispersion_ratio=pct,
        dispersion_ratio_percentile_252d=pct,
    )

    precomputed = build_rule_inputs_by_date(
        features=features,
        spy_close=spy_close,
        realized_vol_percentile_252d=realized_vol_pct,
        vix_percentile_252d=vix_pct,
    )

    for dt in index[20::15]:
        expected = build_rule_inputs_for_date(
            features=features,
            dt=dt,
            spy_close=spy_close,
            realized_vol_percentile_252d=realized_vol_pct,
            vix_percentile_252d=vix_pct,
        )
        actual = precomputed[dt]
        for field in expected.__dataclass_fields__:
            assert getattr(actual, field) == pytest.approx(
                getattr(expected, field), nan_ok=True
            )


def test_rising_fragility_blocked_when_nan_in_trailing_21d_corr_window():
    """N-5: a NaN anywhere in the trailing 21d window of
    `avg_pairwise_corr_63d` must propagate to `avg_pairwise_corr_slope_21d`
    as NaN, which the rule guards against → rising_fragility = False even
    when breadth is weak and the other slope is positive."""
    cfg = _default_rules_config()
    index = pd.bdate_range(end="2024-12-31", periods=50)
    avg_corr = pd.Series(np.linspace(0.30, 0.60, 50), index=index)
    # Inject NaN inside the trailing 21d window (10 sessions before the end).
    avg_corr.iloc[-10] = float("nan")
    rising_eig = pd.Series(np.linspace(0.30, 0.55, 50), index=index)
    flat_pct = pd.Series(0.50, index=index)
    spy_close = pd.Series(np.linspace(400.0, 420.0, 50), index=index)
    eff_rank = pd.Series(4.0, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=avg_corr,
        avg_pairwise_corr_percentile_504d=flat_pct,
        largest_eigenvalue_share=rising_eig,
        largest_eigenvalue_share_percentile_504d=flat_pct,
        effective_rank=eff_rank,
        effective_rank_percentile_504d=flat_pct,
        absorption_ratio_top3=flat_pct,
        dispersion_ratio=flat_pct,
        dispersion_ratio_percentile_252d=flat_pct,
    )
    inputs = build_rule_inputs_for_date(
        features=features,
        dt=index[-1],
        spy_close=spy_close,
        realized_vol_percentile_252d=flat_pct,
        vix_percentile_252d=flat_pct,
    )
    assert math.isnan(inputs.avg_pairwise_corr_slope_21d)
    assert (
        evaluate_rising_fragility(inputs, cfg, breadth_label="weak_breadth") is False
    )


# ---------- Integration over a multi-day features series ---------------------


def test_evaluate_rules_over_multi_day_series_labels_flip_when_thresholds_cross():
    """Construct a NetworkFragilityFeatures series where avg_pairwise_corr
    percentile drifts from 0.50 (diversified_normal) up through 0.78
    (correlation_concentration), then to 0.95 with weak realized-vol so we
    end up at correlation_concentration; with realized_vol high we cross
    into correlation_to_one. Verify the engine flips labels at the
    threshold boundary."""
    cfg = _default_rules_config()
    index = pd.bdate_range(end="2024-12-31", periods=50)

    # avg_corr_pct: 0.50 for days 0..29, 0.80 for 30..39, 0.95 for 40..49.
    avg_corr_pct = pd.Series(
        np.concatenate([
            np.full(30, 0.50),
            np.full(10, 0.80),
            np.full(10, 0.95),
        ]),
        index=index,
    )
    realized_vol_pct = pd.Series(
        np.concatenate([
            np.full(40, 0.50),
            np.full(10, 0.85),   # crosses corr_to_one threshold (>0.80)
        ]),
        index=index,
    )
    flat_50 = pd.Series(0.50, index=index)
    flat_eff_rank = pd.Series(4.0, index=index)
    flat_avg_corr = pd.Series(0.40, index=index)
    flat_eig = pd.Series(0.50, index=index)
    spy_close = pd.Series(np.linspace(400.0, 390.0, 50), index=index)  # drifting down
    vix_pct = flat_50

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=flat_avg_corr,
        avg_pairwise_corr_percentile_504d=avg_corr_pct,
        largest_eigenvalue_share=flat_eig,
        largest_eigenvalue_share_percentile_504d=flat_50,
        effective_rank=flat_eff_rank,
        effective_rank_percentile_504d=flat_50,
        absorption_ratio_top3=flat_50,
        dispersion_ratio=flat_50,
        dispersion_ratio_percentile_252d=flat_50,
    )

    labels_by_day: dict[pd.Timestamp, str] = {}
    for dt in index[21:]:  # need 21d trailing
        inputs = build_rule_inputs_for_date(
            features=features,
            dt=dt,
            spy_close=spy_close,
            realized_vol_percentile_252d=realized_vol_pct,
            vix_percentile_252d=vix_pct,
        )
        labels_by_day[dt] = evaluate_rules(
            inputs=inputs,
            config=cfg,
            breadth_label="healthy_breadth",
            volatility_label="normal_vol",
        )

    # Day 25: avg_corr 0.50, stable, dispersion 0.50 (<0.70) → diversified_normal.
    assert labels_by_day[index[25]] == "diversified_normal"
    # Day 35: avg_corr_pct 0.80 → correlation_concentration (vol still 0.50 so
    # corr_to_one not triggered).
    assert labels_by_day[index[35]] == "correlation_concentration"
    # Day 45: avg_corr_pct 0.95, realized_vol_pct 0.85, drawdown < 0 →
    # correlation_to_one wins.
    assert labels_by_day[index[45]] == "correlation_to_one"


def test_evaluate_rules_end_to_end_via_compute_features():
    """AGENTS rule A: real invocation end-to-end. Build a tiny synthetic
    universe + price series → run compute_features → run the rule engine
    on each session → assert the engine yields a NetworkFragilityLabel
    (not a crash, not unknown for all days)."""
    from regime_detection.fragility_universe import (
        CROSS_ASSET_SYMBOLS,
        INDEX_SYMBOL,
        NETWORK_FRAGILITY_UNIVERSE,
        SECTOR_ETFS,
    )

    cfg = _default_rules_config()
    index = pd.bdate_range(end="2024-12-31", periods=700)
    rng = np.random.default_rng(seed=20260512)
    rets = rng.normal(0.0, 0.01, size=(len(index), len(NETWORK_FRAGILITY_UNIVERSE)))
    prices = pd.DataFrame(
        (1.0 + rets).cumprod(axis=0) * 100.0,
        index=index,
        columns=list(NETWORK_FRAGILITY_UNIVERSE),
    )
    features = compute_features(
        sector_etf_closes={s: prices[s] for s in SECTOR_ETFS},
        cross_asset_closes={s: prices[s] for s in CROSS_ASSET_SYMBOLS},
        spy_close=prices[INDEX_SYMBOL],
    )

    realized_vol_pct = pd.Series(0.40, index=index)
    vix_pct = pd.Series(0.40, index=index)

    seen_labels: set[str] = set()
    for dt in index[-100:]:
        if pd.isna(features.avg_pairwise_corr_percentile_504d.loc[dt]):
            continue
        inputs = build_rule_inputs_for_date(
            features=features,
            dt=dt,
            spy_close=prices[INDEX_SYMBOL],
            realized_vol_percentile_252d=realized_vol_pct,
            vix_percentile_252d=vix_pct,
        )
        label = evaluate_rules(
            inputs=inputs,
            config=cfg,
            breadth_label="neutral_breadth",
            volatility_label="normal_vol",
        )
        seen_labels.add(label)

    # The engine must produce at least one non-unknown label on a 100-day
    # window of real percentile-rank-driven inputs.
    assert seen_labels - {"unknown"}, f"engine yielded only {seen_labels}"


# ---------- Boundary: _trailing_slope / _trailing_stability / _trailing_drawdown --------


@pytest.mark.unit
def test_trailing_slope_returns_nan_when_insufficient_window():
    """_trailing_slope returns NaN when the series has fewer than `window`
    sessions before (inclusive of) `dt`. Line 159 path."""
    from regime_detection.network_fragility_rules import _trailing_slope

    index = pd.bdate_range(end="2024-12-31", periods=5)
    series = pd.Series(np.linspace(0.30, 0.50, 5), index=index)
    # Request a 21-day window — only 5 sessions available.
    result = _trailing_slope(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_slope_returns_nan_when_nan_in_window():
    """_trailing_slope returns NaN when ANY element in the trailing window is NaN.
    This covers the NaN-propagation guard inside _trailing_slope."""
    from regime_detection.network_fragility_rules import _trailing_slope

    index = pd.bdate_range(end="2024-12-31", periods=30)
    series = pd.Series(np.linspace(0.30, 0.60, 30), index=index)
    series.iloc[-5] = float("nan")
    result = _trailing_slope(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_stability_returns_nan_when_insufficient_window():
    """_trailing_stability returns NaN when the series has fewer than `window`
    sessions before (inclusive of) `dt`. Line 178 path."""
    from regime_detection.network_fragility_rules import _trailing_stability

    index = pd.bdate_range(end="2024-12-31", periods=5)
    series = pd.Series(np.linspace(4.0, 5.0, 5), index=index)
    result = _trailing_stability(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_stability_returns_nan_when_nan_in_window():
    """_trailing_stability returns NaN when ANY element in the trailing window
    is NaN. Line 181 path."""
    from regime_detection.network_fragility_rules import _trailing_stability

    index = pd.bdate_range(end="2024-12-31", periods=30)
    series = pd.Series(4.0, index=index, dtype=float)
    series.iloc[-3] = float("nan")
    result = _trailing_stability(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_stability_returns_nan_when_mean_is_zero():
    """_trailing_stability returns NaN when the mean of the window is zero
    (division by zero guard). Line 184 path."""
    from regime_detection.network_fragility_rules import _trailing_stability

    index = pd.bdate_range(end="2024-12-31", periods=30)
    series = pd.Series(0.0, index=index, dtype=float)
    result = _trailing_stability(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_drawdown_returns_nan_when_insufficient_window():
    """_trailing_drawdown returns NaN when the series has fewer than `window`
    sessions before (inclusive of) `dt`. Line 194 path."""
    from regime_detection.network_fragility_rules import _trailing_drawdown

    index = pd.bdate_range(end="2024-12-31", periods=5)
    series = pd.Series(np.linspace(400.0, 420.0, 5), index=index)
    result = _trailing_drawdown(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_drawdown_returns_nan_when_nan_in_window():
    """_trailing_drawdown returns NaN when ANY element in the trailing window
    is NaN. Line 197 path."""
    from regime_detection.network_fragility_rules import _trailing_drawdown

    index = pd.bdate_range(end="2024-12-31", periods=30)
    series = pd.Series(400.0, index=index, dtype=float)
    series.iloc[-5] = float("nan")
    result = _trailing_drawdown(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_trailing_drawdown_returns_nan_when_peak_is_nonpositive():
    """_trailing_drawdown returns NaN when the 21d peak <= 0 (guard line 200)."""
    from regime_detection.network_fragility_rules import _trailing_drawdown

    index = pd.bdate_range(end="2024-12-31", periods=30)
    series = pd.Series(-1.0, index=index, dtype=float)
    result = _trailing_drawdown(series, index[-1], 21)
    assert math.isnan(result)


@pytest.mark.unit
def test_rolling_ols_slope_series_shorter_than_window():
    """_rolling_ols_slope_series returns all-NaN when input is shorter than
    the requested window. Line 249 path."""
    from regime_detection.network_fragility_rules import _rolling_ols_slope_series

    index = pd.bdate_range(end="2024-12-31", periods=5)
    series = pd.Series(np.linspace(0.3, 0.5, 5), index=index)
    result = _rolling_ols_slope_series(series, window=21)
    assert result.isna().all()


@pytest.mark.unit
def test_rolling_stability_series_shorter_than_window():
    """_rolling_stability_series returns all-NaN when input is shorter than
    the requested window. Line 271 path."""
    from regime_detection.network_fragility_rules import _rolling_stability_series

    index = pd.bdate_range(end="2024-12-31", periods=5)
    series = pd.Series(4.0, index=index, dtype=float)
    result = _rolling_stability_series(series, window=21)
    assert result.isna().all()


# ---------- Boundary: build_axis_series guard paths --------------------------


@pytest.mark.unit
def test_build_axis_series_returns_none_when_network_fragility_config_is_none():
    """build_axis_series returns None when context.config.network_fragility is
    None — the config guard at line 577."""
    from unittest.mock import MagicMock

    from regime_detection.network_fragility_rules import build_axis_series

    # Build a minimal feature store with a real feature object.
    feature_store = MagicMock()
    feature_store.network_fragility = MagicMock()  # not None
    context = MagicMock()
    context.config.network_fragility = None  # triggers line 577

    result = build_axis_series(context, feature_store)
    assert result is None


@pytest.mark.unit
def test_build_axis_series_raises_key_error_when_credit_funding_label_missing_session():
    """build_axis_series raises KeyError when credit_funding_active_labels_by_date
    is provided but is missing a required session. Line 651 path.

    The data-quality gate is patched to return a passing quality object so the
    loop reaches the credit_funding check rather than short-circuiting via
    quality_forces_unknown.
    """
    from unittest.mock import MagicMock, patch

    from regime_detection.models import DataQuality
    from regime_detection.network_fragility import NetworkFragilityFeatures
    from regime_detection.network_fragility_rules import build_axis_series

    index = pd.bdate_range(end="2024-12-31", periods=30)
    flat = pd.Series(0.50, index=index)
    eff_rank = pd.Series(4.0, index=index)
    spy_close = pd.Series(400.0, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=flat,
        avg_pairwise_corr_percentile_504d=flat,
        largest_eigenvalue_share=flat,
        largest_eigenvalue_share_percentile_504d=flat,
        effective_rank=eff_rank,
        effective_rank_percentile_504d=flat,
        absorption_ratio_top3=flat,
        dispersion_ratio=flat,
        dispersion_ratio_percentile_252d=flat,
    )

    from regime_detection.config import load_default_regime_config

    cfg = load_default_regime_config()
    nf_cfg = cfg.network_fragility

    feature_store = MagicMock()
    feature_store.network_fragility = features
    feature_store.volatility = MagicMock()
    feature_store.volatility.realized_vol_percentile_252d = flat
    feature_store.volatility.vix_percentile_252d = flat

    sessions = [dt.date() for dt in index[-3:]]
    context = MagicMock()
    context.config = cfg
    context.config.network_fragility = nf_cfg
    context.sessions = sessions
    context.spy_ohlcv = {"close": spy_close}

    # Provide only ONE session in the credit_funding dict — the remaining two
    # sessions in context.sessions are absent → KeyError.
    incomplete_credit_funding = {sessions[0]: "credit_calm"}

    passing_quality = DataQuality(status="ok", freshness_days=1, completeness=1.0)
    with (
        patch(
            "regime_detection.network_fragility_rules.assess_series_input_quality",
            return_value=passing_quality,
        ),
        patch(
            "regime_detection.network_fragility_rules.quality_forces_unknown",
            return_value=False,
        ),
    ):
        with pytest.raises(KeyError):
            build_axis_series(
                context,
                feature_store,
                credit_funding_active_labels_by_date=incomplete_credit_funding,
            )


@pytest.mark.unit
def test_build_axis_series_raises_key_error_when_breadth_label_missing_session():
    """build_axis_series raises KeyError when breadth_active_labels_by_date
    is provided but missing a required session. Line 634 path."""
    from unittest.mock import MagicMock, patch

    from regime_detection.models import DataQuality
    from regime_detection.network_fragility import NetworkFragilityFeatures
    from regime_detection.network_fragility_rules import build_axis_series
    from regime_detection.config import load_default_regime_config

    index = pd.bdate_range(end="2024-12-31", periods=30)
    flat = pd.Series(0.50, index=index)
    spy_close = pd.Series(400.0, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=flat,
        avg_pairwise_corr_percentile_504d=flat,
        largest_eigenvalue_share=flat,
        largest_eigenvalue_share_percentile_504d=flat,
        effective_rank=pd.Series(4.0, index=index),
        effective_rank_percentile_504d=flat,
        absorption_ratio_top3=flat,
        dispersion_ratio=flat,
        dispersion_ratio_percentile_252d=flat,
    )

    cfg = load_default_regime_config()
    feature_store = MagicMock()
    feature_store.network_fragility = features
    feature_store.volatility = MagicMock()
    feature_store.volatility.realized_vol_percentile_252d = flat
    feature_store.volatility.vix_percentile_252d = flat

    sessions = [dt.date() for dt in index[-3:]]
    context = MagicMock()
    context.config = cfg
    context.config.network_fragility = cfg.network_fragility
    context.sessions = sessions
    context.spy_ohlcv = {"close": spy_close}

    # Provide only ONE breadth session — two are missing → KeyError.
    incomplete_breadth = {sessions[0]: "healthy_breadth"}

    passing_quality = DataQuality(status="ok", freshness_days=1, completeness=1.0)
    with (
        patch(
            "regime_detection.network_fragility_rules.assess_series_input_quality",
            return_value=passing_quality,
        ),
        patch(
            "regime_detection.network_fragility_rules.quality_forces_unknown",
            return_value=False,
        ),
    ):
        with pytest.raises(KeyError):
            build_axis_series(
                context,
                feature_store,
                breadth_active_labels_by_date=incomplete_breadth,
            )


@pytest.mark.unit
def test_build_axis_series_raises_key_error_when_volatility_label_missing_session():
    """build_axis_series raises KeyError when volatility_active_labels_by_date
    is provided but missing a required session. Line 643 path."""
    from unittest.mock import MagicMock, patch

    from regime_detection.models import DataQuality
    from regime_detection.network_fragility import NetworkFragilityFeatures
    from regime_detection.network_fragility_rules import build_axis_series
    from regime_detection.config import load_default_regime_config

    index = pd.bdate_range(end="2024-12-31", periods=30)
    flat = pd.Series(0.50, index=index)
    spy_close = pd.Series(400.0, index=index)

    features = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=flat,
        avg_pairwise_corr_percentile_504d=flat,
        largest_eigenvalue_share=flat,
        largest_eigenvalue_share_percentile_504d=flat,
        effective_rank=pd.Series(4.0, index=index),
        effective_rank_percentile_504d=flat,
        absorption_ratio_top3=flat,
        dispersion_ratio=flat,
        dispersion_ratio_percentile_252d=flat,
    )

    cfg = load_default_regime_config()
    feature_store = MagicMock()
    feature_store.network_fragility = features
    feature_store.volatility = MagicMock()
    feature_store.volatility.realized_vol_percentile_252d = flat
    feature_store.volatility.vix_percentile_252d = flat

    sessions = [dt.date() for dt in index[-3:]]
    context = MagicMock()
    context.config = cfg
    context.config.network_fragility = cfg.network_fragility
    context.sessions = sessions
    context.spy_ohlcv = {"close": spy_close}

    # Provide only ONE volatility session — two are missing → KeyError.
    incomplete_volatility = {sessions[0]: "normal_vol"}

    passing_quality = DataQuality(status="ok", freshness_days=1, completeness=1.0)
    with (
        patch(
            "regime_detection.network_fragility_rules.assess_series_input_quality",
            return_value=passing_quality,
        ),
        patch(
            "regime_detection.network_fragility_rules.quality_forces_unknown",
            return_value=False,
        ),
    ):
        with pytest.raises(KeyError):
            build_axis_series(
                context,
                feature_store,
                volatility_active_labels_by_date=incomplete_volatility,
            )
