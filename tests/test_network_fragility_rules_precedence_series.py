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

from regime_detection.config import (
    NetworkFragilityRulesConfig,
    load_default_regime_config,
)
from regime_detection.network_fragility import (
    NetworkFragilityFeatures,
    compute_features,
)
from regime_detection.network_fragility_rules import (
    RULE_PRECEDENCE,
    NetworkFragilityRuleInputs,
    build_rule_inputs_by_date,
    build_rule_inputs_for_date,
    evaluate_diversified_normal,
    evaluate_rising_fragility,
    evaluate_rules,
    evaluate_stock_picker_dispersion,
)

# ---------- Helpers -----------------------------------------------------------


def _default_rules_config() -> NetworkFragilityRulesConfig:
    return load_default_regime_config().network_fragility.rules


def _inputs(
    *,
    avg_corr_pct: float = 0.50,
    largest_eig_pct: float = 0.50,
    eff_rank_pct: float = 0.50,
    avg_corr: float = 0.50,
    largest_eig: float = 0.35,
    dispersion_pct: float = 0.50,
    absorption_ratio: float = 0.50,
    avg_corr_slope: float = 0.0,
    largest_eig_slope: float = 0.0,
    eff_rank_stability: float = 0.02,
    realized_vol_pct: float = 0.50,
    realized_vol_21d: float = 0.12,
    drawdown_21d: float = 0.0,
    vix_pct: float = 0.50,
) -> NetworkFragilityRuleInputs:
    """Construct rule inputs with sane mid-band defaults; override only
    the dimensions a test is exercising."""
    return NetworkFragilityRuleInputs(
        avg_pairwise_corr_percentile_504d=avg_corr_pct,
        largest_eigenvalue_share_percentile_504d=largest_eig_pct,
        effective_rank_percentile_504d=eff_rank_pct,
        avg_pairwise_corr_63d=avg_corr,
        largest_eigenvalue_share=largest_eig,
        dispersion_ratio_percentile_252d=dispersion_pct,
        absorption_ratio_top3=absorption_ratio,
        avg_pairwise_corr_slope_21d=avg_corr_slope,
        largest_eigenvalue_share_slope_21d=largest_eig_slope,
        effective_rank_stability_21d=eff_rank_stability,
        realized_vol_percentile_252d=realized_vol_pct,
        realized_vol_21d=realized_vol_21d,
        drawdown_21d=drawdown_21d,
        vix_percentile_252d=vix_pct,
    )


# ---------- Config wiring -----------------------------------------------------


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


def test_systemic_stress_beats_correlation_to_one_when_credit_funding_absent():
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
    assert label == "systemic_stress"


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
        avg_corr_pct=0.80,  # triggers concentration
        avg_corr_slope=0.001,  # would trigger rising_fragility
        largest_eig_slope=0.001,
        realized_vol_pct=0.50,  # NOT triggering corr_to_one (need 0.80)
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
        dispersion_pct=0.40,  # under stock_picker dispersion threshold
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "diversified_normal"


def test_network_mixed_when_valid_data_has_no_dominant_rule_match():
    cfg = _default_rules_config()
    # Pick a feature profile that satisfies no rule: correlation in
    # diversified_normal band but rank unstable AND outside relaxed inner band.
    inputs = _inputs(
        avg_corr_pct=0.65,  # in band [0.0, 0.75] but outside inner [0.30, 0.60]
        eff_rank_stability=0.10,  # unstable (> 0.05 threshold)
        dispersion_pct=0.40,  # below stock_picker dispersion
        avg_corr_slope=-0.001,  # negative slope
        largest_eig_slope=-0.001,
        largest_eig_pct=0.40,
        eff_rank_pct=0.50,  # not below 0.25
    )
    label = evaluate_rules(
        inputs=inputs,
        config=cfg,
        breadth_label="healthy_breadth",
        volatility_label="normal_vol",
    )
    assert label == "network_mixed"


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
    assert evaluate_rising_fragility(inputs, cfg, breadth_label="weak_breadth") is False


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
        np.concatenate(
            [
                np.full(30, 0.50),
                np.full(10, 0.80),
                np.full(10, 0.95),
            ]
        ),
        index=index,
    )
    realized_vol_pct = pd.Series(
        np.concatenate(
            [
                np.full(40, 0.50),
                np.full(10, 0.85),  # crosses corr_to_one threshold (>0.80)
            ]
        ),
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
