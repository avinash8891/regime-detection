"""Slice 4 — v2 §2C Credit/Funding axis end-to-end tests.

TDD per AGENTS.md / ~/.claude/CLAUDE.md testing rules:
  - Real ticker symbols (HYG, LQD, TLT, KRE, SOFR, IORB, NFCI, broad_usd_index).
  - Real config (load_default_regime_config). No mocks of pandas/fetchers.
  - Hand-computed expected values for numeric assertions.
  - One end-to-end engine test via RegimeEngine.classify.

Spec authority: docs/regime_engine_v2_spec.md §2C lines 2005-2130.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import CreditFundingSeriesClassifier
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    CreditFundingConfig,
    CreditFundingRulesConfig,
    load_default_regime_config,
)
from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CREDIT_SPREAD_PROXY_BIAS_SOURCE,
    CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL,
    CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
    CreditFundingFeatures,
    CreditFundingRuleInputs,
    compute_credit_funding_features,
    evaluate_credit_calm,
    evaluate_credit_stress,
    evaluate_deleveraging,
    evaluate_funding_squeeze,
    evaluate_rules,
    evaluate_spread_widening,
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
from regime_detection.market_context import build_market_context


# --- Synthetic fixtures ------------------------------------------------------

_TRAINING_SESSIONS = 650  # > 504 + 63 cold-start
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260513


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _make_constant_series(index: pd.DatetimeIndex, value: float, name: str) -> pd.Series:
    return pd.Series(value, index=index, name=name)


def _make_random_walk(index: pd.DatetimeIndex, *, seed: int, start: float, sigma: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, size=len(index))
    closes = start * (1.0 + rets).cumprod()
    return pd.Series(closes, index=index, dtype=float)


def _default_rules() -> CreditFundingRulesConfig:
    return load_default_regime_config().credit_funding.rules


# --- Group A — Feature compute (5 tests) -------------------------------------


def test_compute_credit_funding_features_returns_all_series() -> None:
    """All 13 §2C feature series materialise on a 650-session synthetic input."""
    idx = _bdate_index()
    n = len(idx)
    hyg = _make_random_walk(idx, seed=_SEED + 1, start=80.0, sigma=0.005)
    lqd = _make_random_walk(idx, seed=_SEED + 2, start=110.0, sigma=0.003)
    tlt = _make_random_walk(idx, seed=_SEED + 3, start=100.0, sigma=0.006)
    kre = _make_random_walk(idx, seed=_SEED + 4, start=50.0, sigma=0.012)
    spy = _make_random_walk(idx, seed=_SEED + 5, start=400.0, sigma=0.008)
    sofr = _make_constant_series(idx, 5.0, "SOFR")
    iorb = _make_constant_series(idx, 4.9, "IORB")
    # NFCI weekly: assign every 5th index; rest NaN so ffill exercises.
    nfci_w = pd.Series(np.nan, index=idx, name="NFCI", dtype=float)
    rng = np.random.default_rng(_SEED + 6)
    weekly_pos = list(range(0, n, 5))
    nfci_w.iloc[weekly_pos] = rng.normal(-0.5, 0.2, size=len(weekly_pos))
    usd = _make_random_walk(idx, seed=_SEED + 7, start=100.0, sigma=0.003)

    features = compute_credit_funding_features(
        hyg_close=hyg,
        lqd_close=lqd,
        tlt_close=tlt,
        kre_close=kre,
        spy_close=spy,
        sofr=sofr,
        iorb=iorb,
        nfci_weekly=nfci_w,
        broad_usd_index=usd,
        config=_default_rules(),
    )
    for name in features.feature_names:
        s = getattr(features, name)
        assert isinstance(s, pd.Series)
        assert len(s) == n, f"{name} length mismatch: {len(s)} != {n}"


def test_hy_spread_proxy_sign_convention_rising_means_widening() -> None:
    """§2C line 2033: TLT outperforming HYG → positive hy_spread_proxy_63d."""
    idx = _bdate_index(periods=200)
    n = len(idx)
    # Build HYG declining, TLT rising over the test window.
    hyg = pd.Series(np.linspace(100.0, 80.0, n), index=idx, dtype=float)
    tlt = pd.Series(np.linspace(100.0, 130.0, n), index=idx, dtype=float)
    # LQD/KRE/SPY are filler; we only assert on hy_spread_proxy_63d.
    lqd = pd.Series(100.0, index=idx, dtype=float)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    nfci_w.iloc[::5] = -0.3
    usd = pd.Series(100.0, index=idx, dtype=float)

    features = compute_credit_funding_features(
        hyg_close=hyg, lqd_close=lqd, tlt_close=tlt, kre_close=kre,
        spy_close=spy, sofr=sofr, iorb=iorb, nfci_weekly=nfci_w,
        broad_usd_index=usd, config=_default_rules(),
    )
    # At t=100 (well past the 63d window), HYG has fallen, TLT has risen,
    # so tlt_total_return_63d > 0, hyg_total_return_63d < 0, and the
    # differential (TLT - HYG) must be strictly positive.
    val = features.hy_spread_proxy_63d.iloc[100]
    assert val > 0.0, f"expected positive widening proxy, got {val}"


def test_nfci_carries_forward_weekly_to_daily() -> None:
    """§2C line 2049: NFCI weekly→daily forward-fill (last-known-value)."""
    idx = _bdate_index(periods=80)
    n = len(idx)
    hyg = pd.Series(80.0, index=idx, dtype=float)
    lqd = pd.Series(110.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    usd = pd.Series(100.0, index=idx, dtype=float)
    # NFCI weekly: place 4 observations spaced 5 sessions apart starting at idx 10.
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    obs_positions = [10, 15, 20, 25]
    obs_values = [-0.5, -0.3, -0.1, 0.1]
    for pos, val in zip(obs_positions, obs_values):
        nfci_w.iloc[pos] = val

    features = compute_credit_funding_features(
        hyg_close=hyg, lqd_close=lqd, tlt_close=tlt, kre_close=kre,
        spy_close=spy, sofr=sofr, iorb=iorb, nfci_weekly=nfci_w,
        broad_usd_index=usd, config=_default_rules(),
    )
    carried = features.nfci_daily_carried
    # Pre-first-observation: NaN.
    assert pd.isna(carried.iloc[9])
    # Observation rows hold the exact value.
    for pos, expected in zip(obs_positions, obs_values):
        assert carried.iloc[pos] == pytest.approx(expected)
    # Intermediate rows hold the last-known value.
    for inter_pos, expected in [(12, -0.5), (17, -0.3), (22, -0.1), (28, 0.1)]:
        assert carried.iloc[inter_pos] == pytest.approx(expected)


def test_kre_spy_ratio_at_specific_date() -> None:
    """Hand-pinned: KRE=50, SPY=400 → ratio = 50/400 = 0.125."""
    idx = _bdate_index(periods=100)
    n = len(idx)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    # Filler.
    hyg = pd.Series(80.0, index=idx, dtype=float)
    lqd = pd.Series(110.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    nfci_w.iloc[::5] = -0.3
    usd = pd.Series(100.0, index=idx, dtype=float)

    features = compute_credit_funding_features(
        hyg_close=hyg, lqd_close=lqd, tlt_close=tlt, kre_close=kre,
        spy_close=spy, sofr=sofr, iorb=iorb, nfci_weekly=nfci_w,
        broad_usd_index=usd, config=_default_rules(),
    )
    assert features.kre_spy_ratio.iloc[50] == pytest.approx(0.125)


def test_bias_warnings_frame_present_with_expected_code_and_5_rows() -> None:
    """§2C lines 2128-2130: bias-warning row per proxy feature."""
    idx = _bdate_index(periods=200)
    hyg = pd.Series(80.0, index=idx, dtype=float)
    lqd = pd.Series(110.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    nfci_w.iloc[::5] = -0.3
    usd = pd.Series(100.0, index=idx, dtype=float)

    features = compute_credit_funding_features(
        hyg_close=hyg, lqd_close=lqd, tlt_close=tlt, kre_close=kre,
        spy_close=spy, sofr=sofr, iorb=iorb, nfci_weekly=nfci_w,
        broad_usd_index=usd, config=_default_rules(),
    )
    bw = features.bias_warnings
    assert len(bw) == 5
    assert (bw["warning_code"] == CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE).all()
    assert (bw["source"] == CREDIT_SPREAD_PROXY_BIAS_SOURCE).all()
    assert (bw["source_url"] == CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL).all()
    expected_features = {
        "hy_spread_proxy_63d",
        "ig_spread_proxy_63d",
        "hy_spread_proxy_percentile_504d",
        "hy_spread_proxy_slope_21d",
        "ig_spread_proxy_slope_21d",
    }
    assert set(bw["feature_name"]) == expected_features


# --- Group B — Rule precedence (6 tests) -------------------------------------


def _rule_inputs(**overrides: float) -> CreditFundingRuleInputs:
    """Build a CreditFundingRuleInputs with neutral defaults; override per-test."""
    defaults: dict[str, float] = dict(
        hy_spread_proxy_percentile_504d=0.50,
        hy_spread_proxy_slope_21d=0.0,
        ig_spread_proxy_slope_21d=0.0,
        broad_usd_index_zscore_21d=0.0,
        sofr_iorb_slope_21d=0.0,
        spy_21d_return=0.0,
        tlt_21d_return=0.0,
        realized_vol_21d_percentile_252d=0.50,
        avg_pairwise_corr_percentile_504d=0.50,
    )
    defaults.update(overrides)
    return CreditFundingRuleInputs(**defaults)


def test_credit_calm_fires_on_low_percentile_and_non_rising_slope() -> None:
    """§2C lines 2065-2067: pct=0.30 (<0.50) AND slope=-0.001 (≤0)."""
    rules = _default_rules()
    inputs = _rule_inputs(
        hy_spread_proxy_percentile_504d=0.30,
        hy_spread_proxy_slope_21d=-0.001,
    )
    assert evaluate_credit_calm(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "credit_calm"


def test_spread_widening_fires_on_strict_positive_slopes_both_legs() -> None:
    """§2C lines 2069-2071: both HY and IG slopes strictly > 0."""
    rules = _default_rules()
    inputs = _rule_inputs(
        hy_spread_proxy_percentile_504d=0.50,
        hy_spread_proxy_slope_21d=0.002,
        ig_spread_proxy_slope_21d=0.001,
    )
    assert evaluate_spread_widening(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "spread_widening"


def test_credit_stress_fires_on_high_percentile_and_falling_spy() -> None:
    """§2C lines 2073-2075: pct=0.85 AND spy_21d=-0.06."""
    rules = _default_rules()
    inputs = _rule_inputs(
        hy_spread_proxy_percentile_504d=0.85,
        spy_21d_return=-0.06,
    )
    assert evaluate_credit_stress(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "credit_stress"


def test_funding_squeeze_fires_on_usd_zscore_and_sofr_widening_and_falling_spy() -> None:
    """§2C lines 2077-2080: usd_z=2.0, sofr_slope=0.001, spy_21d=-0.02."""
    rules = _default_rules()
    inputs = _rule_inputs(
        broad_usd_index_zscore_21d=2.0,
        sofr_iorb_slope_21d=0.001,
        spy_21d_return=-0.02,
    )
    assert evaluate_funding_squeeze(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "funding_squeeze"


def test_deleveraging_fires_on_5_condition_composite() -> None:
    """§2C lines 2082-2087: spy_21d=-0.07, tlt_21d=-0.01, usd_z=0.5,
    vol_pct=0.80, corr_pct=0.85."""
    rules = _default_rules()
    inputs = _rule_inputs(
        spy_21d_return=-0.07,
        tlt_21d_return=-0.01,
        broad_usd_index_zscore_21d=0.5,
        realized_vol_21d_percentile_252d=0.80,
        avg_pairwise_corr_percentile_504d=0.85,
    )
    assert evaluate_deleveraging(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "deleveraging"


def test_deleveraging_outranks_credit_stress_when_both_match() -> None:
    """§2C line 2019 precedence: deleveraging > credit_stress."""
    rules = _default_rules()
    # Both deleveraging AND credit_stress predicates fire:
    inputs = _rule_inputs(
        # credit_stress: pct > 0.80 AND spy_21d < -0.05
        hy_spread_proxy_percentile_504d=0.85,
        # deleveraging composite (also fires)
        spy_21d_return=-0.07,
        tlt_21d_return=-0.01,
        broad_usd_index_zscore_21d=0.5,
        realized_vol_21d_percentile_252d=0.80,
        avg_pairwise_corr_percentile_504d=0.85,
    )
    assert evaluate_credit_stress(inputs, rules) is True
    assert evaluate_deleveraging(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "deleveraging"


# --- Group C — Unknown gate (4 tests) ----------------------------------------


def _build_full_synthetic_context(
    *,
    hyg_truncate_sessions: int | None = None,
    nfci_truncate_calendar_days: int | None = None,
    sofr_drop_last: bool = False,
):
    """Build a MarketContext with full cross_asset_closes and macro_series.

    Optional knobs simulate spec unknown-gate failure modes.
    """
    idx = _bdate_index(periods=_TRAINING_SESSIONS)
    n = len(idx)
    rng = np.random.default_rng(_SEED)

    # Build full NETWORK_FRAGILITY_UNIVERSE prices (so feature_store.network_fragility lights up).
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
    # Add KRE on cross_asset_closes alongside the §3.1 cross-asset symbols.
    kre_series = _make_random_walk(idx, seed=_SEED + 99, start=50.0, sigma=0.012)
    cross_asset_closes = {s: universe_prices[s] for s in CROSS_ASSET_SYMBOLS}
    cross_asset_closes["KRE"] = kre_series

    # HYG truncation: zero out the last N sessions of HYG to simulate staleness.
    if hyg_truncate_sessions is not None:
        hyg_copy = cross_asset_closes["HYG"].copy()
        hyg_copy.iloc[-hyg_truncate_sessions:] = np.nan
        cross_asset_closes["HYG"] = hyg_copy

    # Macro series — daily SOFR/IORB, weekly NFCI, daily broad_usd_index.
    sofr = _make_constant_series(idx, 5.0, "SOFR")
    iorb = _make_constant_series(idx, 4.9, "IORB")
    if sofr_drop_last:
        sofr = sofr.copy()
        sofr.iloc[-1] = np.nan
    nfci_w = pd.Series(np.nan, index=idx, dtype=float, name="NFCI")
    weekly_positions = list(range(0, n, 5))
    nfci_values = rng.normal(-0.5, 0.2, size=len(weekly_positions))
    for pos, val in zip(weekly_positions, nfci_values):
        nfci_w.iloc[pos] = val
    if nfci_truncate_calendar_days is not None:
        # Wipe NFCI for the last `nfci_truncate_calendar_days` calendar days.
        cutoff = idx[-1] - pd.Timedelta(days=nfci_truncate_calendar_days)
        nfci_w.loc[nfci_w.index > cutoff] = np.nan
    usd = _make_random_walk(idx, seed=_SEED + 100, start=100.0, sigma=0.003)

    macro_series = {
        "SOFR": sofr, "IORB": iorb, "NFCI": nfci_w,
        "broad_usd_index": usd,
        # Add yield series for monetary slice compatibility.
        "DGS2": _make_constant_series(idx, 4.5, "DGS2"),
        "DGS10": _make_constant_series(idx, 4.0, "DGS10"),
    }

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


def _build_store_and_outputs(context):
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    return store, CreditFundingSeriesClassifier().build(context, store)


def test_unknown_when_hyg_stale_more_than_5_sessions() -> None:
    """§2C line 2123: HYG stale > 5 sessions → unknown gate trip."""
    context = _build_full_synthetic_context(hyg_truncate_sessions=10)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "etf_stale:HYG" in (out.data_quality.reason or "")


def test_unknown_when_nfci_stale_more_than_14_days() -> None:
    """§2C line 2124: NFCI stale > 14 calendar days → unknown gate trip."""
    context = _build_full_synthetic_context(nfci_truncate_calendar_days=20)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "nfci_stale" in (out.data_quality.reason or "")


def test_unknown_when_sofr_missing() -> None:
    """§2C line 2125: SOFR missing at session → unknown gate trip."""
    context = _build_full_synthetic_context(sofr_drop_last=True)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "sofr_missing" in (out.data_quality.reason or "")


def test_unknown_when_assess_series_input_quality_fails() -> None:
    """§2C line 2126: assess_series_input_quality fails → unknown.

    Forced by mutating the feature store so the spread-proxy series is all NaN
    (insufficient history) — staleness gate passes because the underlying ETF
    closes are intact, so the secondary quality gate must be what catches us.
    """
    context = _build_full_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
    )
    cf = store.credit_funding
    assert cf is not None
    nan_series = pd.Series(np.nan, index=cf.hy_spread_proxy_63d.index)
    broken = CreditFundingFeatures(
        hy_spread_proxy_63d=nan_series,
        ig_spread_proxy_63d=nan_series,
        hy_spread_proxy_percentile_504d=nan_series,
        hy_spread_proxy_slope_21d=nan_series,
        ig_spread_proxy_slope_21d=nan_series,
        kre_spy_ratio=nan_series,
        kre_spy_slope_63d=nan_series,
        nfci_daily_carried=nan_series,
        sofr_iorb_spread=nan_series,
        sofr_iorb_slope_21d=nan_series,
        broad_usd_index_zscore_21d=nan_series,
        spy_21d_return=nan_series,
        tlt_21d_return=nan_series,
        bias_warnings=cf.bias_warnings,
    )
    broken_store = store.model_copy(update={"credit_funding": broken})
    outputs = CreditFundingSeriesClassifier().build(context, broken_store)
    assert outputs is not None
    last_day = context.sessions[-1]
    assert outputs[last_day].raw_label == "unknown"


# --- Group D — Hysteresis (2 tests) ------------------------------------------


def test_deleveraging_holds_for_5_deescalation_days() -> None:
    """§2C lines 2111-2117: deleveraging→credit_calm transitions held 5d."""
    deesc = load_default_regime_config().credit_funding.deescalation_days_by_label
    # 10 days deleveraging, then switch to credit_calm. Hold period = 5.
    raws = ["deleveraging"] * 10 + ["credit_calm"] * 10
    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    # Stable still deleveraging on the first 4 post-flip days (positions 10..13).
    for i in range(10, 14):
        assert stable[i] == "deleveraging", f"position {i}: {stable[i]}"
    # On position 14 (the 5th post-flip day) the hold expires; per the
    # hysteresis implementation, pending_count >= threshold triggers the flip.
    assert stable[14] == "credit_calm"


def test_credit_calm_deescalates_immediately() -> None:
    """§2C line 2115: credit_calm holds 0 days (immediate de-escalation)."""
    deesc = load_default_regime_config().credit_funding.deescalation_days_by_label
    # Start in credit_calm (rank 0), flip to spread_widening (rank 1).
    # spread_widening has HIGHER risk_rank — escalation must be immediate.
    raws = ["credit_calm"] * 5 + ["spread_widening"] * 5
    stable, _active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    # Immediate escalation: position 5 must already be spread_widening.
    assert stable[5] == "spread_widening"


# --- Group E — Wire integration (3 tests) ------------------------------------


def test_feature_store_credit_funding_seam_none_without_kre_in_cross_asset_closes() -> None:
    """Missing KRE on cross_asset_closes → feature_store.credit_funding is None."""
    context = _build_full_synthetic_context()
    # Strip KRE from the cross_asset_closes dict.
    stripped = {k: v for k, v in (context.cross_asset_closes or {}).items() if k != "KRE"}
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
        new_context, credit_funding_config=new_context.config.credit_funding
    )
    assert store.credit_funding is None


def test_feature_store_credit_funding_seam_lit_with_all_inputs() -> None:
    """All 8 §2C inputs present → feature_store.credit_funding is populated."""
    context = _build_full_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
    )
    assert store.credit_funding is not None
    assert isinstance(store.credit_funding, CreditFundingFeatures)


def test_regime_output_carries_credit_funding_state_when_configured() -> None:
    """End-to-end: classify_window populates RegimeOutput.credit_funding_state."""
    context = _build_full_synthetic_context()
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
    assert out.credit_funding_state is not None
    allowed = set(CREDIT_FUNDING_RISK_RANK.keys())
    assert out.credit_funding_state.active_label in allowed
