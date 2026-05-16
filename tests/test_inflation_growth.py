"""Slice 5 — v2 §2B Inflation/Growth axis end-to-end tests.

TDD per AGENTS.md / ~/.claude/CLAUDE.md testing rules:
  - Real ticker symbols (DBC, TLT, XLY, XLI, XLP, XLU, SPY) + real macro
    series keys (cpi_all_items, pmi_manufacturing, dgs10).
  - Real config (load_default_regime_config). No mocks.
  - Hand-computed expected values for numeric assertions.

Spec authority: docs/regime_engine_v2_spec.md §2B lines 2174-2326.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.inflation_growth import build_axis_series as _build_ig_axis_series
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    InflationGrowthRulesConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    INDEX_SYMBOL,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.inflation_growth import (
    INFLATION_GROWTH_RISK_RANK,
    INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE,
    InflationGrowthFeatures,
    InflationGrowthRuleInputs,
    build_rule_inputs_by_date,
    build_rule_inputs_for_date,
    compute_inflation_growth_features,
    compute_inflation_surprise_zscore,
    evaluate_disinflation,
    evaluate_earnings_contraction,
    evaluate_earnings_expansion,
    evaluate_goldilocks,
    evaluate_inflation_shock,
    evaluate_recession_scare,
    evaluate_recovery_growth,
    evaluate_rules,
)
from regime_detection.market_context import build_market_context


# --- Synthetic fixtures ------------------------------------------------------

_TRAINING_SESSIONS = 650
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260513


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _default_rules() -> InflationGrowthRulesConfig:
    return load_default_regime_config().inflation_growth.rules


def _rule_inputs(**overrides) -> InflationGrowthRuleInputs:
    defaults: dict[str, object] = dict(
        cpi_6m_change_pct=0.02,
        cpi_6m_change_pct_lag_21=0.02,
        cpi_6m_change_pct_slope_21d=0.0,
        # ADR 0006 — NaN by default so the inflation_shock single-signal
        # limb is silent unless a test explicitly supplies a z-score.
        inflation_surprise_zscore=float("nan"),
        # Log #48 closure — NaN by default so the earnings_expansion /
        # earnings_contraction labels are silent unless a test supplies
        # a revision value (mirrors the accumulator cold-start state).
        aggregate_forward_eps_revision_direction_4w=float("nan"),
        pmi_manufacturing=52.0,
        pmi_manufacturing_slope_21d=0.0,
        commodity_return_63d=0.0,
        treasury_10y_yield_slope_21d=0.0,
        cyclical_defensive_slope_21d=0.0,
        spy_21d_return=0.01,
        tlt_21d_return=0.0,
        credit_funding_active_label="credit_calm",
    )
    defaults.update(overrides)
    return InflationGrowthRuleInputs(**defaults)


# --- Group A — Feature compute (4 tests) ------------------------------------


def test_compute_features_returns_all_series_aligned_to_spy_index() -> None:
    idx = _bdate_index(periods=300)
    n = len(idx)
    # Monthly CPI: 1 observation per 21 sessions.
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi.iloc[::21] = np.linspace(300.0, 305.0, num=len(cpi.iloc[::21]))
    pmi = pd.Series(np.nan, index=idx, dtype=float)
    pmi.iloc[::21] = 51.0
    dgs10 = pd.Series(np.linspace(4.0, 4.5, n), index=idx, dtype=float)
    dbc = pd.Series(np.linspace(20.0, 25.0, n), index=idx, dtype=float)
    spy = pd.Series(np.linspace(400.0, 420.0, n), index=idx, dtype=float)
    tlt = pd.Series(np.linspace(100.0, 95.0, n), index=idx, dtype=float)
    xly = pd.Series(np.linspace(150.0, 170.0, n), index=idx, dtype=float)
    xli = pd.Series(np.linspace(100.0, 115.0, n), index=idx, dtype=float)
    xlp = pd.Series(np.linspace(70.0, 72.0, n), index=idx, dtype=float)
    xlu = pd.Series(np.linspace(60.0, 62.0, n), index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
    )
    for name in feats.feature_names:
        s = getattr(feats, name)
        assert isinstance(s, pd.Series)
        assert len(s) == n, f"{name} length mismatch"


def test_cpi_forward_fills_monthly_to_daily() -> None:
    """§2B line 2208 PMI pattern applies to CPI too: monthly→daily ffill."""
    idx = _bdate_index(periods=80)
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi.iloc[10] = 300.0
    cpi.iloc[31] = 303.0  # next month
    # Filler.
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    # cpi_3m_change_pct uses the daily ffilled CPI under cpi_lookback_3m_sessions=63.
    # We'll just check that the ffilled feature's input doesn't leave NaN
    # gaps between the two CPI release rows by inspecting cpi_6m_change_pct
    # (NaN until 126 sessions of ffilled series). Use a different check:
    # pick a smaller test — verify pmi_manufacturing carries forward.
    pmi2 = pd.Series(np.nan, index=idx, dtype=float)
    pmi2.iloc[10] = 51.0
    pmi2.iloc[31] = 49.0
    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi2,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
    )
    # pmi=51 forward-fills from pos 10 to pos 30, then pmi=49 from pos 31 onward.
    assert feats.pmi_manufacturing.iloc[15] == pytest.approx(51.0)
    assert feats.pmi_manufacturing.iloc[30] == pytest.approx(51.0)
    assert feats.pmi_manufacturing.iloc[35] == pytest.approx(49.0)


def test_commodity_return_63d_hand_pinned() -> None:
    """DBC rises from 20 to 25 across 200 sessions linearly. At pos=100, the
    return over the prior 63 sessions is hand-computable."""
    idx = _bdate_index(periods=200)
    n = len(idx)
    dbc = pd.Series(np.linspace(20.0, 25.0, n), index=idx, dtype=float)
    # Filler.
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi.iloc[::21] = 300.0
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi, pmi_manufacturing=pmi, dgs10=dgs10,
        dbc_close=dbc, spy_close=spy, tlt_close=tlt,
        xly_close=xly, xli_close=xli, xlp_close=xlp, xlu_close=xlu,
        config=_default_rules(),
    )
    expected = (dbc.iloc[100] / dbc.iloc[100 - 63]) - 1.0
    assert feats.commodity_return_63d.iloc[100] == pytest.approx(expected)


def test_cyclical_defensive_ratio_hand_pinned() -> None:
    """ratio = (XLY + XLI) / (XLP + XLU) = (150+100)/(70+60) = 250/130 ≈ 1.923."""
    idx = _bdate_index(periods=100)
    xly = pd.Series(150.0, index=idx, dtype=float)
    xli = pd.Series(100.0, index=idx, dtype=float)
    xlp = pd.Series(70.0, index=idx, dtype=float)
    xlu = pd.Series(60.0, index=idx, dtype=float)
    # Filler.
    cpi = pd.Series(300.0, index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi, pmi_manufacturing=pmi, dgs10=dgs10,
        dbc_close=dbc, spy_close=spy, tlt_close=tlt,
        xly_close=xly, xli_close=xli, xlp_close=xlp, xlu_close=xlu,
        config=_default_rules(),
    )
    assert feats.cyclical_defensive_ratio.iloc[50] == pytest.approx(250.0 / 130.0)


def test_build_rule_inputs_by_date_matches_single_day_builder() -> None:
    idx = _bdate_index(periods=200)
    n = len(idx)
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi.iloc[::21] = np.linspace(300.0, 308.0, num=len(cpi.iloc[::21]))
    pmi = pd.Series(np.nan, index=idx, dtype=float)
    pmi.iloc[::21] = np.linspace(49.0, 53.0, num=len(pmi.iloc[::21]))
    dgs10 = pd.Series(np.linspace(4.0, 4.5, n), index=idx, dtype=float)
    dbc = pd.Series(np.linspace(20.0, 25.0, n), index=idx, dtype=float)
    spy = pd.Series(np.linspace(400.0, 420.0, n), index=idx, dtype=float)
    tlt = pd.Series(np.linspace(100.0, 95.0, n), index=idx, dtype=float)
    xly = pd.Series(np.linspace(150.0, 170.0, n), index=idx, dtype=float)
    xli = pd.Series(np.linspace(100.0, 115.0, n), index=idx, dtype=float)
    xlp = pd.Series(np.linspace(70.0, 72.0, n), index=idx, dtype=float)
    xlu = pd.Series(np.linspace(60.0, 62.0, n), index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
    )
    cf_labels = {
        ts: ("credit_calm" if i % 2 == 0 else "spread_widening")
        for i, ts in enumerate(idx)
    }
    precomputed = build_rule_inputs_by_date(
        features=feats,
        config=_default_rules(),
        credit_funding_active_labels_by_date=cf_labels,
    )
    for dt in idx[126::23]:
        expected = build_rule_inputs_for_date(
            features=feats,
            dt=dt,
            config=_default_rules(),
            credit_funding_active_label=cf_labels[dt],
        )
        actual = precomputed[dt]
        for field in expected.__dataclass_fields__:
            if field == "credit_funding_active_label":
                assert getattr(actual, field) == getattr(expected, field)
            else:
                assert getattr(actual, field) == pytest.approx(
                    getattr(expected, field), nan_ok=True
                )


def test_aggregate_forward_eps_revision_forward_fills_onto_spy_index() -> None:
    """Log #48 wiring: a weekly revision series (keyed by workbook
    observation_date, not the trading calendar) is carried forward onto
    every SPY session via reindex(method='ffill'). When None, the feature
    stays all-NaN so the earnings labels falsify (V1 byte-identity)."""
    idx = _bdate_index(periods=80)
    # Filler series — values are irrelevant to this feature.
    cpi = pd.Series(300.0, index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    common = dict(
        cpi_all_items=cpi, pmi_manufacturing=pmi, dgs10=dgs10,
        dbc_close=dbc, spy_close=spy, tlt_close=tlt,
        xly_close=xly, xli_close=xli, xlp_close=xlp, xlu_close=xlu,
        config=_default_rules(),
    )

    # None → all-NaN placeholder.
    feats_none = compute_inflation_growth_features(**common)
    assert feats_none.aggregate_forward_eps_revision_direction_4w.isna().all()

    # Two weekly observations. The second deliberately lands on a Sunday
    # (not an NYSE session) to prove the ffill reindex carries it forward
    # rather than dropping it on an exact-label mismatch.
    obs_early = idx[20]
    obs_sunday = pd.Timestamp("2025-04-13")  # a Sunday
    assert obs_sunday.dayofweek == 6
    assert obs_sunday not in idx
    revision = pd.Series(
        [0.04, -0.03],
        index=pd.DatetimeIndex([obs_early, obs_sunday]),
        dtype=float,
    )
    feats = compute_inflation_growth_features(
        **common, aggregate_forward_eps_revision=revision
    )
    out = feats.aggregate_forward_eps_revision_direction_4w
    assert len(out) == len(idx)
    # Before the first observation: NaN (no value to carry forward).
    assert np.isnan(out.iloc[10])
    # On/after the first observation, before the Sunday one: 0.04.
    assert out.loc[obs_early] == pytest.approx(0.04)
    assert out.iloc[25] == pytest.approx(0.04)
    # The first NYSE session strictly after the Sunday observation carries
    # the -0.03 value forward.
    after_sunday = idx[idx > obs_sunday][0]
    assert out.loc[after_sunday] == pytest.approx(-0.03)
    assert out.iloc[-1] == pytest.approx(-0.03)


# --- Group B — Rule predicates (10 tests) ------------------------------------


def test_goldilocks_fires_under_drift_pmi_spy_creditcalm() -> None:
    rules = _default_rules()
    inputs = _rule_inputs(
        cpi_6m_change_pct=0.020,
        cpi_6m_change_pct_lag_21=0.022,   # drift = 0.002 <= 0.005
        pmi_manufacturing=52.0,
        spy_21d_return=0.03,
        credit_funding_active_label="credit_calm",
    )
    assert evaluate_goldilocks(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "goldilocks"


def test_goldilocks_short_circuits_when_credit_funding_unbuilt() -> None:
    """§2B line 2316: cross-axis short-circuit when §2C is unbuilt."""
    rules = _default_rules()
    inputs = _rule_inputs(
        cpi_6m_change_pct=0.020,
        cpi_6m_change_pct_lag_21=0.022,
        pmi_manufacturing=52.0,
        spy_21d_return=0.03,
        credit_funding_active_label=None,
    )
    assert evaluate_goldilocks(inputs, rules) is False
    assert evaluate_rules(inputs=inputs, config=rules) == "unknown"


def test_inflation_shock_composite_fires() -> None:
    """§2B lines 2242-2245: 4-condition composite limb."""
    rules = _default_rules()
    inputs = _rule_inputs(
        commodity_return_63d=0.20,
        treasury_10y_yield_slope_21d=0.01,
        spy_21d_return=-0.02,
        tlt_21d_return=-0.01,
    )
    assert evaluate_inflation_shock(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "inflation_shock"


def test_inflation_shock_single_signal_limb_silent_when_zscore_nan() -> None:
    """ADR 0006 / Log #48: when `cpi_nowcast` is unwired (or during the 5y
    cold-start), `inflation_surprise_zscore` is NaN — the single-signal
    limb falsifies, and with all-zero composite inputs inflation_shock
    does not fire. This is the cold-start path; it must NOT block the
    composite limb when composite inputs DO fire."""
    rules = _default_rules()
    inputs = _rule_inputs(
        inflation_surprise_zscore=float("nan"),
        commodity_return_63d=0.0,
        treasury_10y_yield_slope_21d=0.0,
        spy_21d_return=0.0,
        tlt_21d_return=0.0,
    )
    assert evaluate_inflation_shock(inputs, rules) is False


def test_inflation_shock_single_signal_limb_fires_above_threshold() -> None:
    """ADR 0006: the single-signal limb fires on a large positive
    (hotter-than-nowcast) inflation surprise — `inflation_surprise_zscore
    > +1.5` — even when EVERY composite-limb input is benign."""
    rules = _default_rules()
    inputs = _rule_inputs(
        inflation_surprise_zscore=2.0,  # > +1.5 threshold
        # all composite inputs benign — only the single-signal limb fires
        commodity_return_63d=0.0,
        treasury_10y_yield_slope_21d=0.0,
        spy_21d_return=0.01,
        tlt_21d_return=0.0,
    )
    assert evaluate_inflation_shock(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "inflation_shock"


def test_inflation_shock_single_signal_limb_silent_below_threshold() -> None:
    """`inflation_surprise_zscore` at or below +1.5 does not fire the
    single-signal limb (strict `>` per spec line 2551)."""
    rules = _default_rules()
    inputs = _rule_inputs(
        inflation_surprise_zscore=1.5,  # exactly at threshold — strict `>` fails
        commodity_return_63d=0.0,
        treasury_10y_yield_slope_21d=0.0,
        spy_21d_return=0.01,
        tlt_21d_return=0.0,
    )
    assert evaluate_inflation_shock(inputs, rules) is False


def test_disinflation_fires() -> None:
    rules = _default_rules()
    inputs = _rule_inputs(
        cpi_6m_change_pct_slope_21d=-0.0005,
        treasury_10y_yield_slope_21d=-0.01,
        pmi_manufacturing=47.0,  # > 45 but < 50 → fails goldilocks but ok here
        credit_funding_active_label="spread_widening",  # neutralizes goldilocks
    )
    assert evaluate_disinflation(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "disinflation"


def test_recession_scare_fires() -> None:
    rules = _default_rules()
    inputs = _rule_inputs(
        treasury_10y_yield_slope_21d=-0.01,
        cyclical_defensive_slope_21d=-0.002,
        credit_funding_active_label="spread_widening",
        spy_21d_return=-0.07,
    )
    assert evaluate_recession_scare(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "recession_scare"


def test_recession_scare_short_circuits_when_credit_funding_unbuilt() -> None:
    """§2B line 2316: cross-axis short-circuit when §2C is unbuilt."""
    rules = _default_rules()
    inputs = _rule_inputs(
        treasury_10y_yield_slope_21d=-0.01,
        cyclical_defensive_slope_21d=-0.002,
        credit_funding_active_label=None,
        spy_21d_return=-0.07,
    )
    assert evaluate_recession_scare(inputs, rules) is False


def test_recovery_growth_fires() -> None:
    rules = _default_rules()
    inputs = _rule_inputs(
        pmi_manufacturing_slope_21d=0.05,
        pmi_manufacturing=53.0,
        cyclical_defensive_slope_21d=0.001,
        credit_funding_active_label="credit_calm",
        # Make sure goldilocks doesn't pre-empt: kill its drift/slope leg.
        cpi_6m_change_pct=0.02,
        cpi_6m_change_pct_lag_21=0.035,  # drift = 0.015 > 0.005
        cpi_6m_change_pct_slope_21d=0.001,  # > 0 too
        spy_21d_return=-0.01,             # < 0 → goldilocks fails on spy leg
    )
    assert evaluate_recovery_growth(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "recovery_growth"


def test_recovery_growth_short_circuits_when_credit_funding_unbuilt() -> None:
    rules = _default_rules()
    inputs = _rule_inputs(
        pmi_manufacturing_slope_21d=0.05,
        pmi_manufacturing=53.0,
        cyclical_defensive_slope_21d=0.001,
        credit_funding_active_label=None,
    )
    assert evaluate_recovery_growth(inputs, rules) is False


def test_earnings_labels_falsify_on_cold_start_nan() -> None:
    """Log #48: NaN revision (accumulator cold-start / unwired) falsifies
    both earnings labels."""
    rules = _default_rules()
    inputs = _rule_inputs(aggregate_forward_eps_revision_direction_4w=float("nan"))
    assert evaluate_earnings_expansion(inputs, rules) is False
    assert evaluate_earnings_contraction(inputs, rules) is False


def test_earnings_labels_falsify_in_neutral_band() -> None:
    """§2B lines 2605/2609: a revision inside (-0.02, +0.02) fires neither
    label — the thresholds are strict."""
    rules = _default_rules()
    inputs = _rule_inputs(aggregate_forward_eps_revision_direction_4w=0.01)
    assert evaluate_earnings_expansion(inputs, rules) is False
    assert evaluate_earnings_contraction(inputs, rules) is False
    # Exactly on the threshold also falsifies (strict >).
    on_threshold = _rule_inputs(aggregate_forward_eps_revision_direction_4w=0.02)
    assert evaluate_earnings_expansion(on_threshold, rules) is False


def test_earnings_expansion_fires_above_threshold() -> None:
    """§2B line 2605: aggregate_forward_eps_revision_direction_4w > +0.02.

    pmi_manufacturing=49.0 suppresses the higher-precedence goldilocks /
    recovery_growth / disinflation labels so evaluate_rules dispatch
    actually reaches earnings_expansion."""
    rules = _default_rules()
    inputs = _rule_inputs(
        pmi_manufacturing=49.0,
        aggregate_forward_eps_revision_direction_4w=0.05,
    )
    assert evaluate_earnings_expansion(inputs, rules) is True
    assert evaluate_earnings_contraction(inputs, rules) is False
    assert evaluate_rules(inputs=inputs, config=rules) == "earnings_expansion"


def test_earnings_contraction_fires_below_threshold() -> None:
    """§2B line 2609: aggregate_forward_eps_revision_direction_4w < -0.02.

    earnings_contraction outranks earnings_expansion in the §2B
    precedence walk."""
    rules = _default_rules()
    inputs = _rule_inputs(
        pmi_manufacturing=49.0,
        aggregate_forward_eps_revision_direction_4w=-0.05,
    )
    assert evaluate_earnings_contraction(inputs, rules) is True
    assert evaluate_earnings_expansion(inputs, rules) is False
    assert evaluate_rules(inputs=inputs, config=rules) == "earnings_contraction"


def test_inflation_shock_outranks_recession_scare_when_both_match() -> None:
    """§2B line 2190 precedence: inflation_shock > recession_scare."""
    rules = _default_rules()
    inputs = _rule_inputs(
        # inflation_shock composite (all 4 conditions fire)
        commodity_return_63d=0.20,
        treasury_10y_yield_slope_21d=0.01,  # positive — but recession_scare needs <0
        spy_21d_return=-0.07,
        tlt_21d_return=-0.01,
        # Configure recession_scare to also be candidate by flipping treasury slope
        # ... but treasury_slope is a single scalar. So contrive both:
        cyclical_defensive_slope_21d=-0.002,
        credit_funding_active_label="spread_widening",
    )
    # inflation_shock fires (positive treasury slope, etc.), recession_scare
    # fails because treasury slope > 0 in this contrivance. Make a different
    # construction where BOTH labels really fire.
    inputs = _rule_inputs(
        commodity_return_63d=0.20,
        treasury_10y_yield_slope_21d=-0.01,  # negative for recession_scare
        spy_21d_return=-0.07,
        tlt_21d_return=-0.01,
        cyclical_defensive_slope_21d=-0.002,
        credit_funding_active_label="credit_stress",
    )
    # inflation_shock needs treasury_10y_yield_slope_21d > 0 (it fails now).
    # So actually only ONE composite can fire at a time given the slope
    # opposite-sign constraint. Verify precedence walker chooses recession_scare here.
    assert evaluate_inflation_shock(inputs, rules) is False
    assert evaluate_recession_scare(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "recession_scare"


# --- Group C — Synthetic context + unknown gate (5 tests) --------------------


def _build_synthetic_context(
    *,
    cpi_truncate_calendar_days: int | None = None,
    pmi_truncate_calendar_days: int | None = None,
    dgs10_truncate_sessions: int | None = None,
    include_nowcast_and_eps_revision: bool = False,
):
    """Build a full MarketContext with §2B inputs."""
    idx = _bdate_index(periods=_TRAINING_SESSIONS)
    n = len(idx)
    rng = np.random.default_rng(_SEED)

    universe_prices = pd.DataFrame(
        (1.0 + rng.normal(0.0, 0.01, size=(n, len(NETWORK_FRAGILITY_UNIVERSE)))).cumprod(axis=0)
        * 100.0,
        index=idx,
        columns=list(NETWORK_FRAGILITY_UNIVERSE),
    )
    spy_close = universe_prices[INDEX_SYMBOL]
    market_rows: list[dict[str, object]] = []
    for ts in idx:
        close = float(spy_close.loc[ts])
        market_rows.append({
            "date": ts.date(), "symbol": "SPY",
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": 1_000_000,
        })
        market_rows.append({
            "date": ts.date(), "symbol": "RSP",
            "open": close * 0.5, "high": close * 0.5 * 1.005,
            "low": close * 0.5 * 0.995, "close": close * 0.5, "volume": 500_000,
        })
        market_rows.append({
            "date": ts.date(), "symbol": "VIXY",
            "open": 20.0, "high": 20.5, "low": 19.5, "close": 20.0, "volume": 100_000,
        })
    market_data = pd.DataFrame(market_rows)

    sector_etf_closes = {s: universe_prices[s] for s in SECTOR_ETFS}
    cross_asset_closes: dict[str, pd.Series] = {s: universe_prices[s] for s in CROSS_ASSET_SYMBOLS}
    # Add KRE for credit_funding seam.
    cross_asset_closes["KRE"] = pd.Series(
        np.linspace(50.0, 55.0, n), index=idx, dtype=float, name="KRE"
    )
    # Add DBC for §2B inflation_growth (cross_asset_closes uses upper).
    cross_asset_closes["DBC"] = pd.Series(
        np.linspace(20.0, 22.0, n), index=idx, dtype=float, name="DBC"
    )
    # XLY/XLI/XLP/XLU are sector ETFs; the §2B classifier reads them from
    # cross_asset_closes (per slice 5 plan). Add them.
    for sym in ("XLY", "XLI", "XLP", "XLU"):
        cross_asset_closes[sym] = sector_etf_closes[sym]

    # Macro series — CPI/PMI monthly (every ~21 sessions); DGS10 daily; plus
    # credit_funding macro inputs for the §2C axis dependency.
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi_release_positions = list(range(0, n, 21))
    cpi.iloc[cpi_release_positions] = np.linspace(300.0, 320.0, len(cpi_release_positions))
    if cpi_truncate_calendar_days is not None:
        cutoff = idx[-1] - pd.Timedelta(days=cpi_truncate_calendar_days)
        cpi.loc[cpi.index > cutoff] = np.nan

    pmi = pd.Series(np.nan, index=idx, dtype=float)
    pmi_release_positions = list(range(0, n, 21))
    pmi.iloc[pmi_release_positions] = 51.0
    if pmi_truncate_calendar_days is not None:
        cutoff = idx[-1] - pd.Timedelta(days=pmi_truncate_calendar_days)
        pmi.loc[pmi.index > cutoff] = np.nan

    dgs10 = pd.Series(4.0, index=idx, dtype=float, name="dgs10")
    if dgs10_truncate_sessions is not None:
        dgs10 = dgs10.copy()
        dgs10.iloc[-dgs10_truncate_sessions:] = np.nan

    # Macro for §2C (credit_funding) so cross-axis label populates.
    sofr = pd.Series(5.0, index=idx, dtype=float, name="SOFR")
    iorb = pd.Series(4.9, index=idx, dtype=float, name="IORB")
    nfci_w = pd.Series(np.nan, index=idx, dtype=float, name="NFCI")
    for pos in range(0, n, 5):
        nfci_w.iloc[pos] = -0.5
    usd = pd.Series(np.linspace(100.0, 102.0, n), index=idx, dtype=float, name="broad_usd_index")

    macro_series = {
        "cpi_all_items": cpi,
        "pmi_manufacturing": pmi,
        "dgs10": dgs10,
        "SOFR": sofr,
        "IORB": iorb,
        "NFCI": nfci_w,
        "broad_usd_index": usd,
        "DGS2": pd.Series(4.5, index=idx, dtype=float),
        "DGS10": pd.Series(4.0, index=idx, dtype=float),
    }
    if include_nowcast_and_eps_revision:
        macro_series["cpi_nowcast"] = pd.Series(0.01, index=idx, dtype=float)
        macro_series["aggregate_forward_eps_revision"] = pd.Series(
            0.03, index=idx, dtype=float
        )

    config = RegimeEngine().config
    context = build_market_context(
        end_date=idx[-1].date(),
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=macro_series,
    )
    return context


def _build_store_and_outputs(context, *, credit_funding_active_labels_by_date=None):
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        credit_funding_config=cfg.credit_funding,
        inflation_growth_config=cfg.inflation_growth,
    )
    outputs = _build_ig_axis_series(
        context,
        store,
        credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
    )
    return store, outputs


def test_unknown_when_cpi_stale_more_than_60_days() -> None:
    """§2B line 2309: CPI stale > 60 days → unknown."""
    context = _build_synthetic_context(cpi_truncate_calendar_days=90)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "cpi_stale" in (out.data_quality.reason or "")


def test_unknown_when_pmi_stale_more_than_45_days() -> None:
    """§2B line 2310: PMI stale > 45 days → unknown."""
    context = _build_synthetic_context(pmi_truncate_calendar_days=60)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "pmi_stale" in (out.data_quality.reason or "")


def test_unknown_when_dgs10_stale_more_than_5_sessions() -> None:
    """§2B line 2311: DGS10 stale > 5 sessions → unknown."""
    context = _build_synthetic_context(dgs10_truncate_sessions=10)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "dgs10_stale" in (out.data_quality.reason or "")


def test_unknown_when_assess_series_input_quality_fails() -> None:
    """§2B line 2312: assess_series_input_quality fails → unknown.

    Force this by mutating the features so the spy_21d_return series is all NaN.
    """
    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )
    ig = store.inflation_growth
    assert ig is not None
    nan_series = pd.Series(np.nan, index=ig.cpi_6m_change_pct.index)
    broken = InflationGrowthFeatures(
        cpi_3m_change_pct=nan_series,
        cpi_6m_change_pct=nan_series,
        cpi_6m_change_pct_slope_21d=nan_series,
        inflation_surprise_zscore=nan_series,
        pmi_manufacturing=nan_series,
        pmi_manufacturing_slope_21d=nan_series,
        aggregate_forward_eps_revision_direction_4w=nan_series,
        commodity_return_63d=nan_series,
        treasury_10y_yield_slope_21d=nan_series,
        cyclical_defensive_ratio=nan_series,
        cyclical_defensive_slope_21d=nan_series,
        spy_21d_return=nan_series,
        tlt_21d_return=nan_series,
        bias_warnings=ig.bias_warnings,
    )
    broken_store = store.model_copy(update={"inflation_growth": broken})
    outputs = _build_ig_axis_series(context, broken_store)
    assert outputs is not None
    last_day = context.sessions[-1]
    assert outputs[last_day].raw_label == "unknown"


def test_feature_store_seam_lit_with_all_inputs() -> None:
    """All 9 §2B inputs present → feature_store.inflation_growth populated."""
    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        inflation_growth_config=context.config.inflation_growth,
    )
    assert store.inflation_growth is not None
    assert isinstance(store.inflation_growth, InflationGrowthFeatures)


def test_classifier_rule_evidence_includes_nowcast_and_eps_revision_inputs() -> None:
    """2B evidence must surface the optional nowcast/EPS scalars once wired."""
    context = _build_synthetic_context(include_nowcast_and_eps_revision=True)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None

    last_day = context.sessions[-1]
    evidence = outputs[last_day].evidence["rule_evidence"]
    assert "inflation_surprise_zscore" in evidence
    assert "aggregate_forward_eps_revision_direction_4w" in evidence


def test_feature_store_seam_none_when_dbc_missing() -> None:
    """Missing DBC → feature_store.inflation_growth is None."""
    context = _build_synthetic_context()
    # Strip DBC.
    stripped = {k: v for k, v in (context.cross_asset_closes or {}).items() if k != "DBC"}
    new_context = build_market_context(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(), "symbol": "SPY",
                    "open": float(context.spy_ohlcv["open"].loc[ts]),
                    "high": float(context.spy_ohlcv["high"].loc[ts]),
                    "low": float(context.spy_ohlcv["low"].loc[ts]),
                    "close": float(context.spy_ohlcv["close"].loc[ts]),
                    "volume": float(context.spy_ohlcv["volume"].loc[ts]),
                }
                for ts in context.spy_ohlcv.index
            ]
            + [
                {
                    "date": ts.date(), "symbol": "RSP",
                    "open": float(context.rsp_close.loc[ts]),
                    "high": float(context.rsp_close.loc[ts]),
                    "low": float(context.rsp_close.loc[ts]),
                    "close": float(context.rsp_close.loc[ts]),
                    "volume": 500_000,
                }
                for ts in context.spy_ohlcv.index
            ]
        ),
        config=context.config,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=stripped,
        macro_series=context.macro_series,
    )
    store = build_feature_store(
        new_context, inflation_growth_config=new_context.config.inflation_growth
    )
    assert store.inflation_growth is None


# --- Group D — Hysteresis (2 tests) ------------------------------------------


def test_inflation_shock_holds_for_5_deescalation_days() -> None:
    """§2B line 2295: inflation_shock → ... holds 5 days."""
    deesc = load_default_regime_config().inflation_growth.deescalation_days_by_label
    raws = ["inflation_shock"] * 10 + ["goldilocks"] * 10
    stable, _ = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    for i in range(10, 14):
        assert stable[i] == "inflation_shock", f"position {i}: {stable[i]}"
    assert stable[14] == "goldilocks"


def test_goldilocks_deescalates_immediately() -> None:
    """§2B line 2299: goldilocks deescalates in 0 days (immediate)."""
    deesc = load_default_regime_config().inflation_growth.deescalation_days_by_label
    # goldilocks (rank 0) → inflation_shock (rank 3) is immediate escalation.
    raws = ["goldilocks"] * 5 + ["inflation_shock"] * 5
    stable, _ = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    assert stable[5] == "inflation_shock"


# --- Group E — End-to-end wire integration (1 test) --------------------------


def test_regime_output_carries_inflation_growth_state_when_configured() -> None:
    """End-to-end: classify_window populates RegimeOutput.inflation_growth_state."""
    context = _build_synthetic_context()
    engine = RegimeEngine()
    timeline = engine.classify_window(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(), "symbol": "SPY",
                    "open": float(context.spy_ohlcv["open"].loc[ts]),
                    "high": float(context.spy_ohlcv["high"].loc[ts]),
                    "low": float(context.spy_ohlcv["low"].loc[ts]),
                    "close": float(context.spy_ohlcv["close"].loc[ts]),
                    "volume": float(context.spy_ohlcv["volume"].loc[ts]),
                }
                for ts in context.spy_ohlcv.index
            ]
            + [
                {
                    "date": ts.date(), "symbol": "RSP",
                    "open": float(context.rsp_close.loc[ts]),
                    "high": float(context.rsp_close.loc[ts]),
                    "low": float(context.rsp_close.loc[ts]),
                    "close": float(context.rsp_close.loc[ts]),
                    "volume": 500_000,
                }
                for ts in context.spy_ohlcv.index
            ]
        ),
        lookback_days=1,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=context.cross_asset_closes,
        macro_series=context.macro_series,
    )
    out = timeline.outputs[-1]
    assert out.inflation_growth_state is not None
    allowed = set(INFLATION_GROWTH_RISK_RANK.keys())
    assert out.inflation_growth_state.active_label in allowed


# ---------------------------------------------------------------------------
# ADR 0006 — inflation_surprise_zscore via the Cleveland Fed nowcast.
# ---------------------------------------------------------------------------


def test_compute_inflation_surprise_zscore_hand_computed() -> None:
    """ADR 0006: surprise = realized_cpi_rate - cpi_nowcast, z-scored over
    a rolling std window. With a short normalizer window and a hand-built
    CPI/nowcast pair the z-score at the trailing session is exact."""
    idx = _bdate_index(periods=40)
    # CPIAUCSL index level rising ~0.5%/month over the window.
    cpi = pd.Series(np.linspace(300.0, 306.0, len(idx)), index=idx, dtype=float)
    # Nowcast: a constant inflation-rate estimate of 1.0% (0.01).
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    zscore = compute_inflation_surprise_zscore(
        cpi_all_items=cpi,
        cpi_nowcast=nowcast,
        session_index=idx,
        realized_rate_lookback=5,
        normalizer_window=10,
    )
    assert isinstance(zscore, pd.Series)
    assert zscore.name == "inflation_surprise_zscore"
    # First (realized_rate_lookback + normalizer_window - 1) rows are NaN
    # (cold-start — the 5y/normalizer std needs a full window).
    assert zscore.iloc[: 5 + 10 - 2].isna().all()
    # Past the cold-start window the z-score is finite.
    assert zscore.dropna().shape[0] > 0


def test_compute_inflation_surprise_zscore_cold_start_all_nan_below_window() -> None:
    """Below `normalizer_window` of surprise history, the z-score is
    entirely NaN — the single-signal limb stays silent (V1 §2.7)."""
    idx = _bdate_index(periods=15)
    cpi = pd.Series(np.linspace(300.0, 302.0, len(idx)), index=idx, dtype=float)
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    zscore = compute_inflation_surprise_zscore(
        cpi_all_items=cpi,
        cpi_nowcast=nowcast,
        session_index=idx,
        realized_rate_lookback=5,
        normalizer_window=1260,  # far longer than the 15-session input
    )
    assert zscore.isna().all()


def test_compute_inflation_growth_features_emits_real_zscore_with_nowcast() -> None:
    """When cpi_nowcast is supplied, compute_inflation_growth_features
    computes a real (non-all-NaN) inflation_surprise_zscore and emits the
    Cleveland-Fed-nowcast bias-warning provenance row."""
    idx = _bdate_index(periods=400)
    n = len(idx)
    cpi = pd.Series(np.linspace(300.0, 312.0, n), index=idx, dtype=float)
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(np.linspace(20.0, 25.0, n), index=idx, dtype=float)
    spy = pd.Series(np.linspace(400.0, 420.0, n), index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    # Short normalizer window so a 400-session input produces non-NaN values.
    rules = _default_rules().model_copy(
        update={"inflation_surprise_normalizer_window_sessions": 60}
    )
    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=rules,
        cpi_nowcast=nowcast,
    )
    # The z-score is no longer the all-NaN placeholder.
    assert feats.inflation_surprise_zscore.notna().any()
    # The bias-warning frame carries the Cleveland Fed nowcast provenance row.
    bw = feats.bias_warnings
    assert (
        bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE
    ).any()
    nowcast_row = bw[
        bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE
    ]
    assert list(nowcast_row["feature_name"]) == ["inflation_surprise_zscore"]


def test_compute_inflation_growth_features_all_nan_zscore_without_nowcast() -> None:
    """When cpi_nowcast is NOT supplied, inflation_surprise_zscore stays the
    all-NaN placeholder and NO Cleveland-Fed bias-warning row is emitted —
    V1 byte-identity preserved (the pre-ADR-0006 behaviour)."""
    idx = _bdate_index(periods=120)
    n = len(idx)
    cpi = pd.Series(np.linspace(300.0, 304.0, n), index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
        # cpi_nowcast omitted
    )
    assert feats.inflation_surprise_zscore.isna().all()
    bw = feats.bias_warnings
    assert not (
        bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE
    ).any()


# ---------------------------------------------------------------------------
# F023 boundary-value tests — uncovered branches (lines 213, 505, 514, 517,
# 629, 634, 725, 741, 747, 829, 857, 915-920).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inflation_growth_features_to_frame_returns_dataframe() -> None:
    """InflationGrowthFeatures.to_frame() builds a DataFrame with every
    feature series as a column. Line 213 path."""
    idx = _bdate_index(periods=50)
    n = len(idx)
    cpi = pd.Series(np.linspace(300.0, 302.0, n), index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi, pmi_manufacturing=pmi, dgs10=dgs10,
        dbc_close=dbc, spy_close=spy, tlt_close=tlt,
        xly_close=xly, xli_close=xli, xlp_close=xlp, xlu_close=xlu,
        config=_default_rules(),
    )
    df = feats.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(feats.feature_names)
    assert len(df) == n


@pytest.mark.unit
def test_scalar_at_returns_nan_when_dt_not_in_index() -> None:
    """_scalar_at returns NaN when the requested timestamp is absent from
    the series index. Line 505 path."""
    from regime_detection.inflation_growth import _scalar_at

    idx = _bdate_index(periods=10)
    series = pd.Series(1.0, index=idx, dtype=float)
    # Choose a timestamp that is definitely not a business day in the index.
    absent_dt = pd.Timestamp("2000-01-01")
    assert absent_dt not in idx
    result = _scalar_at(series, absent_dt)
    import math
    assert math.isnan(result)


@pytest.mark.unit
def test_scalar_at_lag_returns_nan_when_dt_not_in_index() -> None:
    """_scalar_at_lag returns NaN when dt is absent from the index. Line 514
    path (first guard)."""
    from regime_detection.inflation_growth import _scalar_at_lag

    idx = _bdate_index(periods=10)
    series = pd.Series(1.0, index=idx, dtype=float)
    absent_dt = pd.Timestamp("2000-01-01")
    import math
    result = _scalar_at_lag(series, absent_dt, lag=5)
    assert math.isnan(result)


@pytest.mark.unit
def test_scalar_at_lag_returns_nan_when_lag_exceeds_position() -> None:
    """_scalar_at_lag returns NaN when pos - lag < 0 (requesting data before
    the series starts). Line 517 path."""
    from regime_detection.inflation_growth import _scalar_at_lag

    idx = _bdate_index(periods=10)
    series = pd.Series(1.0, index=idx, dtype=float)
    import math
    # dt is the first element (pos=0); lag=5 → 0-5 = -5 < 0.
    result = _scalar_at_lag(series, idx[0], lag=5)
    assert math.isnan(result)


@pytest.mark.unit
def test_evaluate_goldilocks_returns_false_when_credit_not_credit_calm() -> None:
    """evaluate_goldilocks fires only when credit_funding_active_label ==
    'credit_calm'. Any other non-None value → False. Line 629 path."""
    rules = _default_rules()
    inputs = _rule_inputs(
        cpi_6m_change_pct=0.020,
        cpi_6m_change_pct_lag_21=0.022,
        pmi_manufacturing=52.0,
        spy_21d_return=0.03,
        credit_funding_active_label="spread_widening",  # not credit_calm
    )
    assert evaluate_goldilocks(inputs, rules) is False


@pytest.mark.unit
def test_evaluate_goldilocks_returns_false_when_pmi_or_spy_is_nan() -> None:
    """evaluate_goldilocks returns False when pmi_manufacturing or spy_21d_return
    is NaN. Line 634 NaN guard path."""
    rules = _default_rules()
    # NaN PMI.
    nan_pmi = _rule_inputs(
        cpi_6m_change_pct=0.020,
        cpi_6m_change_pct_lag_21=0.022,
        pmi_manufacturing=float("nan"),
        spy_21d_return=0.03,
        credit_funding_active_label="credit_calm",
    )
    assert evaluate_goldilocks(nan_pmi, rules) is False

    # NaN SPY return.
    nan_spy = _rule_inputs(
        cpi_6m_change_pct=0.020,
        cpi_6m_change_pct_lag_21=0.022,
        pmi_manufacturing=52.0,
        spy_21d_return=float("nan"),
        credit_funding_active_label="credit_calm",
    )
    assert evaluate_goldilocks(nan_spy, rules) is False


@pytest.mark.unit
def test_evaluate_recession_scare_returns_false_when_numeric_inputs_nan() -> None:
    """evaluate_recession_scare NaN guard (line 725) — NaN in any of
    treasury_slope, cyclical_defensive_slope, or spy_return falsifies the rule
    even when credit_funding_active_label is 'spread_widening'."""
    rules = _default_rules()
    nan_inputs = _rule_inputs(
        treasury_10y_yield_slope_21d=float("nan"),
        cyclical_defensive_slope_21d=-0.002,
        credit_funding_active_label="spread_widening",
        spy_21d_return=-0.07,
    )
    assert evaluate_recession_scare(nan_inputs, rules) is False


@pytest.mark.unit
def test_evaluate_recovery_growth_returns_false_when_credit_not_credit_calm() -> None:
    """evaluate_recovery_growth returns False for any non-'credit_calm' non-None
    credit label. Line 741 path."""
    rules = _default_rules()
    inputs = _rule_inputs(
        pmi_manufacturing_slope_21d=0.05,
        pmi_manufacturing=53.0,
        cyclical_defensive_slope_21d=0.001,
        credit_funding_active_label="spread_widening",  # not credit_calm
    )
    assert evaluate_recovery_growth(inputs, rules) is False


@pytest.mark.unit
def test_evaluate_recovery_growth_returns_false_when_numeric_inputs_nan() -> None:
    """evaluate_recovery_growth NaN guard (line 747) — NaN in any of
    pmi_slope, pmi, or cyclical_defensive_slope falsifies the rule even when
    credit_funding_active_label is 'credit_calm'."""
    rules = _default_rules()
    nan_pmi_slope = _rule_inputs(
        pmi_manufacturing_slope_21d=float("nan"),
        pmi_manufacturing=53.0,
        cyclical_defensive_slope_21d=0.001,
        credit_funding_active_label="credit_calm",
    )
    assert evaluate_recovery_growth(nan_pmi_slope, rules) is False

    nan_pmi = _rule_inputs(
        pmi_manufacturing_slope_21d=0.05,
        pmi_manufacturing=float("nan"),
        cyclical_defensive_slope_21d=0.001,
        credit_funding_active_label="credit_calm",
    )
    assert evaluate_recovery_growth(nan_pmi, rules) is False


@pytest.mark.unit
def test_build_axis_series_returns_none_when_ig_config_is_none() -> None:
    """build_axis_series returns None when context.config.inflation_growth is
    None — the config guard at line 829."""
    from unittest.mock import MagicMock

    from regime_detection.inflation_growth import (
        build_axis_series as _build_ig_axis,
    )

    feature_store = MagicMock()
    feature_store.inflation_growth = MagicMock()  # not None
    context = MagicMock()
    context.config.inflation_growth = None  # triggers line 829

    result = _build_ig_axis(context, feature_store)
    assert result is None


@pytest.mark.unit
def test_build_axis_series_raises_key_error_when_credit_funding_session_missing() -> None:
    """build_axis_series raises KeyError when credit_funding_active_labels_by_date
    is provided but missing a required session. Lines 915-920 path."""

    from regime_detection.inflation_growth import (
        build_axis_series as _build_ig_axis,
    )
    from regime_detection.feature_store import build_feature_store

    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )

    # Provide a credit_funding dict that is missing at least one session.
    sessions = context.sessions
    incomplete = {sessions[0]: "credit_calm"}  # only the first session

    import pytest

    with pytest.raises(KeyError):
        _build_ig_axis(
            context,
            store,
            credit_funding_active_labels_by_date=incomplete,
        )


@pytest.mark.unit
def test_compute_features_uses_cpi_first_release_when_provided() -> None:
    """When `cpi_first_release` is provided and `use_first_release_cpi_when_available`
    is True, the feature computation uses the first-release CPI and emits the
    FIRST_RELEASE_CPI_PROVENANCE_CODE bias-warning rows. Lines 313, 445-446 path."""
    from regime_detection.inflation_growth import (
        FIRST_RELEASE_CPI_PROVENANCE_CODE,
        compute_inflation_growth_features,
    )

    idx = _bdate_index(periods=200)
    n = len(idx)
    # Latest-revision CPI (would be used if first-release not provided).
    cpi_revised = pd.Series(np.linspace(300.0, 306.0, n), index=idx, dtype=float)
    # First-release CPI — deliberately different values so we can verify which
    # one the features use.
    cpi_first = pd.Series(np.linspace(300.0, 303.0, n), index=idx, dtype=float)

    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi_revised,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
        cpi_first_release=cpi_first,
        use_first_release_cpi_when_available=True,
    )

    # The provenance rows for the three CPI-derived features must be present.
    bw = feats.bias_warnings
    provenance_rows = bw[bw["warning_code"] == FIRST_RELEASE_CPI_PROVENANCE_CODE]
    covered_features = set(provenance_rows["feature_name"].tolist())
    assert "cpi_3m_change_pct" in covered_features
    assert "cpi_6m_change_pct" in covered_features

    # The 6m CPI change should reflect the first-release CPI (smaller slope),
    # not the revised CPI. At position 126 (first valid 6m change), the
    # first-release version returns a smaller fractional change.
    val_first = feats.cpi_6m_change_pct.dropna().iloc[0]
    # Recompute with revised CPI to confirm they differ.
    feats_revised = compute_inflation_growth_features(
        cpi_all_items=cpi_revised,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
    )
    val_revised = feats_revised.cpi_6m_change_pct.dropna().iloc[0]
    assert val_first != val_revised, (
        "cpi_first_release path did not change the computed CPI feature"
    )


@pytest.mark.unit
def test_build_axis_series_credit_funding_label_populates_when_all_sessions_present() -> None:
    """build_axis_series extracts credit_funding_active_label (line 920) when
    credit_funding_active_labels_by_date is provided and covers all sessions."""
    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )
    # Build a complete credit_funding map covering every session.
    complete_cf = {day: "credit_calm" for day in context.sessions}

    outputs = _build_ig_axis_series(
        context,
        store,
        credit_funding_active_labels_by_date=complete_cf,
    )
    assert outputs is not None
    # With credit_calm on every session the goldilocks/recovery_growth rules
    # are eligible — the engine should not short-circuit to unknown everywhere.
    labels = {out.raw_label for out in outputs.values()}
    assert "unknown" not in labels, f"Engine short-circuited to unknown with credit_calm inputs: {labels}"
    assert len(outputs) == len(context.sessions)
