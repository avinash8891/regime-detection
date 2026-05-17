"""Slice 4 — v2 §2C Credit/Funding axis end-to-end tests.

TDD per AGENTS.md / ~/.claude/CLAUDE.md testing rules:
  - Real ticker symbols (HYG, LQD, TLT, KRE, SOFR, IORB, NFCI, broad_usd_index).
  - Real config (load_default_regime_config). No mocks of pandas/fetchers.
  - Hand-computed expected values for numeric assertions.
  - One end-to-end engine test via RegimeEngine.classify.

Spec authority: docs/regime_engine_v2_spec.md §2C lines 2005-2130.
"""

from __future__ import annotations


from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import (
    build_credit_funding_axis_series,
    build_credit_funding_proxy_axis_series,
    resolve_credit_funding_effective_output,
)
from regime_detection.models import CreditFundingOutput, DataQuality
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    CreditFundingRulesConfig,
    load_default_regime_config,
)
from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
    CREDIT_SPREAD_SOURCE,
    CREDIT_SPREAD_SOURCE_CODE,
    CREDIT_SPREAD_SOURCE_URL,
    CreditFundingFeatures,
    CreditFundingRuleInputs,
    build_rule_inputs_by_date,
    build_rule_inputs_for_date,
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
_REAL_FIXTURE_CREDIT_AS_OF = date(2026, 5, 12)


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _make_constant_series(
    index: pd.DatetimeIndex, value: float, name: str
) -> pd.Series:
    return pd.Series(value, index=index, name=name)


def _make_random_walk(
    index: pd.DatetimeIndex, *, seed: int, start: float, sigma: float
) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, size=len(index))
    closes = start * (1.0 + rets).cumprod()
    return pd.Series(closes, index=index, dtype=float)


def _default_rules() -> CreditFundingRulesConfig:
    return load_default_regime_config().credit_funding.rules


# --- Group A — Feature compute (5 tests) -------------------------------------


def test_compute_credit_funding_features_returns_all_series() -> None:
    """All 18 §2C feature series materialise on a 650-session synthetic input."""
    idx = _bdate_index()
    n = len(idx)
    hyg = _make_random_walk(idx, seed=_SEED + 1, start=80.0, sigma=0.008)
    lqd = _make_random_walk(idx, seed=_SEED + 2, start=110.0, sigma=0.006)
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
    # ICE BofA OAS series — single source for the §2C spread metric.
    hy_oas = _make_random_walk(idx, seed=_SEED + 8, start=400.0, sigma=0.01)
    ig_oas = _make_random_walk(idx, seed=_SEED + 9, start=150.0, sigma=0.01)

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
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )
    for name in features.feature_names:
        s = getattr(features, name)
        assert isinstance(s, pd.Series)
        assert len(s) == n, f"{name} length mismatch: {len(s)} != {n}"


def test_hy_oas_63d_carries_hy_oas_input_verbatim() -> None:
    """Single-source contract: `hy_oas_63d` IS the `hy_oas` input
    (ICE BofA OAS), reindexed to the SPY calendar. The §2C line 2033 sign
    convention — rising series = widening spread — holds trivially because
    a rising OAS literally IS a widening spread."""
    idx = _bdate_index(periods=200)
    n = len(idx)
    # Widening HY OAS: 350bps → 600bps over the window.
    hy_oas = pd.Series(np.linspace(350.0, 600.0, n), index=idx, dtype=float)
    ig_oas = pd.Series(np.linspace(120.0, 180.0, n), index=idx, dtype=float)
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
        hyg_close=hyg,
        lqd_close=lqd,
        tlt_close=tlt,
        kre_close=kre,
        spy_close=spy,
        sofr=sofr,
        iorb=iorb,
        nfci_weekly=nfci_w,
        broad_usd_index=usd,
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )
    # hy_oas_63d carries the OAS series verbatim (reindexed only).
    pd.testing.assert_series_equal(
        features.hy_oas_63d,
        hy_oas.rename("hy_oas_63d"),
        check_names=True,
    )
    # Sign convention holds: a later (wider-OAS) session has a higher value.
    assert features.hy_oas_63d.iloc[150] > features.hy_oas_63d.iloc[50]


def test_nfci_carries_forward_weekly_to_daily() -> None:
    """§2C line 2049: NFCI weekly→daily forward-fill (last-known-value)."""
    idx = _bdate_index(periods=80)
    hyg = pd.Series(80.0, index=idx, dtype=float)
    lqd = pd.Series(110.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    usd = pd.Series(100.0, index=idx, dtype=float)
    hy_oas = pd.Series(400.0, index=idx, dtype=float)
    ig_oas = pd.Series(150.0, index=idx, dtype=float)
    # NFCI weekly: place 4 observations spaced 5 sessions apart starting at idx 10.
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    obs_positions = [10, 15, 20, 25]
    obs_values = [-0.5, -0.3, -0.1, 0.1]
    for pos, val in zip(obs_positions, obs_values):
        nfci_w.iloc[pos] = val

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
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
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
    hy_oas = pd.Series(400.0, index=idx, dtype=float)
    ig_oas = pd.Series(150.0, index=idx, dtype=float)

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
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )
    assert features.kre_spy_ratio.iloc[50] == pytest.approx(0.125)


def test_credit_spread_provenance_row_present_with_single_source_code() -> None:
    """§2C credit-spread provenance: 5 OAS rows (ICE BofA via FRED) + 5
    proxy rows (TLT-vs-HYG/LQD total-return-differential) in bias_warnings."""
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
    hy_oas = pd.Series(400.0, index=idx, dtype=float)
    ig_oas = pd.Series(150.0, index=idx, dtype=float)

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
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )
    bw = features.bias_warnings
    # 5 OAS provenance rows + 5 proxy provenance rows.
    assert len(bw) == 10
    oas_rows = bw[bw["warning_code"] == CREDIT_SPREAD_SOURCE_CODE]
    assert len(oas_rows) == 5
    assert (oas_rows["source"] == CREDIT_SPREAD_SOURCE).all()
    assert (oas_rows["source_url"] == CREDIT_SPREAD_SOURCE_URL).all()
    expected_oas_features = {
        "hy_oas_63d",
        "ig_oas_63d",
        "hy_oas_percentile_504d",
        "hy_oas_slope_21d",
        "ig_oas_slope_21d",
    }
    assert set(oas_rows["feature_name"]) == expected_oas_features
    proxy_rows = bw[bw["warning_code"] == CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE]
    assert len(proxy_rows) == 5
    expected_proxy_features = {
        "hy_tr_differential_63d",
        "ig_tr_differential_63d",
        "hy_tr_differential_percentile_504d",
        "hy_tr_differential_slope_21d",
        "ig_tr_differential_slope_21d",
    }
    assert set(proxy_rows["feature_name"]) == expected_proxy_features


def test_ice_bofa_oas_is_the_single_credit_spread_source() -> None:
    """Single-source contract (Log #49 closure): `hy_oas` / `ig_oas` (the
    FRED-redistributed ICE BofA OAS series) populate the
    `hy_oas_63d` / `ig_oas_63d` columns verbatim. There
    is no proxy fallback — both inputs are required positional kwargs."""
    idx = _bdate_index(periods=200)
    n = len(idx)
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

    # Real OAS series in basis-point units (rising = wider spread).
    hy_oas = pd.Series(np.linspace(300.0, 500.0, n), index=idx, dtype=float)
    ig_oas = pd.Series(np.linspace(100.0, 200.0, n), index=idx, dtype=float)

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
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )

    # The spread columns carry the OAS inputs verbatim (reindexed only).
    pd.testing.assert_series_equal(
        features.hy_oas_63d,
        hy_oas.rename("hy_oas_63d"),
        check_names=True,
    )
    pd.testing.assert_series_equal(
        features.ig_oas_63d,
        ig_oas.rename("ig_oas_63d"),
        check_names=True,
    )

    # OAS features carry the single-source ICE-BofA-OAS-via-FRED code.
    bw = features.bias_warnings
    oas_rows = bw[bw["warning_code"] == CREDIT_SPREAD_SOURCE_CODE]
    assert len(oas_rows) == 5


def test_compute_credit_funding_features_requires_both_oas_series() -> None:
    """No proxy fallback: `hy_oas` and `ig_oas` are required kwargs.
    Omitting either raises TypeError at the call boundary — there is no
    silent half-real / half-proxy path."""
    idx = _bdate_index(periods=50)
    hyg = pd.Series(80.0, index=idx, dtype=float)
    lqd = pd.Series(110.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    kre = pd.Series(50.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    sofr = pd.Series(5.0, index=idx, dtype=float)
    iorb = pd.Series(4.9, index=idx, dtype=float)
    nfci_w = pd.Series(np.nan, index=idx, dtype=float)
    usd = pd.Series(100.0, index=idx, dtype=float)
    hy_oas = pd.Series(400.0, index=idx, dtype=float)

    with pytest.raises(TypeError):
        compute_credit_funding_features(
            hyg_close=hyg,
            lqd_close=lqd,
            tlt_close=tlt,
            kre_close=kre,
            spy_close=spy,
            sofr=sofr,
            iorb=iorb,
            nfci_weekly=nfci_w,
            broad_usd_index=usd,
            hy_oas=hy_oas,  # ig_oas omitted — required kwarg
            config=_default_rules(),
        )


def test_tlt_proxy_differential_hand_computed() -> None:
    """hy_tr_differential_63d = tlt_total_return_63d - hyg_total_return_63d,
    with total_return_lookback_days = 63 (config default)."""
    idx = _bdate_index(periods=300)
    hyg = _make_random_walk(idx, seed=11, start=80.0, sigma=0.004)
    lqd = _make_random_walk(idx, seed=12, start=110.0, sigma=0.003)
    tlt = _make_random_walk(idx, seed=13, start=95.0, sigma=0.005)
    kre = _make_random_walk(idx, seed=14, start=55.0, sigma=0.006)
    spy = _make_random_walk(idx, seed=15, start=420.0, sigma=0.008)
    hy_oas = _make_constant_series(idx, 3.5, "hy_oas")
    ig_oas = _make_constant_series(idx, 1.2, "ig_oas")
    sofr = _make_constant_series(idx, 5.3, "sofr")
    iorb = _make_constant_series(idx, 5.4, "iorb")
    nfci = _make_constant_series(idx, -0.2, "nfci")
    usd = _make_random_walk(idx, seed=16, start=120.0, sigma=0.003)

    features = compute_credit_funding_features(
        hyg_close=hyg,
        lqd_close=lqd,
        tlt_close=tlt,
        kre_close=kre,
        spy_close=spy,
        sofr=sofr,
        iorb=iorb,
        nfci_weekly=nfci,
        broad_usd_index=usd,
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )

    w = _default_rules().total_return_lookback_days  # 63
    t = 200
    tlt_tr = tlt.iloc[t] / tlt.iloc[t - w] - 1.0
    hyg_tr = hyg.iloc[t] / hyg.iloc[t - w] - 1.0
    assert features.hy_tr_differential_63d.iloc[t] == pytest.approx(tlt_tr - hyg_tr)
    # ig leg: same total-return-differential formula on LQD
    lqd_tr = lqd.iloc[t] / lqd.iloc[t - w] - 1.0
    assert features.ig_tr_differential_63d.iloc[t] == pytest.approx(tlt_tr - lqd_tr)


def test_tlt_proxy_features_carry_proxy_bias_warning() -> None:
    """The five hy_tr_differential_* / ig_tr_differential_* features each
    carry a credit_spread_proxy_total_return_differential bias-warning row;
    the hy_oas_* features keep the credit_spread_ice_bofa_oas_fred row."""
    idx = _bdate_index(periods=300)
    common = dict(
        hyg_close=_make_constant_series(idx, 80.0, "HYG"),
        lqd_close=_make_constant_series(idx, 110.0, "LQD"),
        tlt_close=_make_constant_series(idx, 95.0, "TLT"),
        kre_close=_make_constant_series(idx, 55.0, "KRE"),
        spy_close=_make_constant_series(idx, 420.0, "SPY"),
        sofr=_make_constant_series(idx, 5.3, "sofr"),
        iorb=_make_constant_series(idx, 5.4, "iorb"),
        nfci_weekly=_make_constant_series(idx, -0.2, "nfci"),
        broad_usd_index=_make_constant_series(idx, 120.0, "usd"),
        hy_oas=_make_constant_series(idx, 3.5, "hy_oas"),
        ig_oas=_make_constant_series(idx, 1.2, "ig_oas"),
        config=_default_rules(),
    )
    features = compute_credit_funding_features(**common)
    codes = set(
        zip(
            features.bias_warnings["feature_name"],
            features.bias_warnings["warning_code"],
        )
    )
    assert (
        "hy_tr_differential_63d",
        "credit_spread_proxy_total_return_differential",
    ) in codes
    assert ("hy_oas_63d", "credit_spread_ice_bofa_oas_fred") in codes


def test_build_rule_inputs_by_date_matches_single_day_builder() -> None:
    idx = _bdate_index(periods=650)
    n = len(idx)
    hyg = pd.Series(np.linspace(75.0, 85.0, n), index=idx, dtype=float)
    lqd = pd.Series(np.linspace(105.0, 115.0, n), index=idx, dtype=float)
    tlt = pd.Series(np.linspace(95.0, 109.0, n), index=idx, dtype=float)
    kre = pd.Series(np.linspace(40.0, 43.0, n), index=idx, dtype=float)
    spy = pd.Series(np.linspace(400.0, 430.0, n), index=idx, dtype=float)
    sofr = pd.Series(np.linspace(5.0, 5.4, n), index=idx, dtype=float)
    iorb = pd.Series(np.linspace(4.9, 5.0, n), index=idx, dtype=float)
    nfci = pd.Series(np.linspace(0.1, 0.5, n), index=idx, dtype=float)
    usd = pd.Series(np.linspace(100.0, 106.0, n), index=idx, dtype=float)
    hy_oas = pd.Series(np.linspace(350.0, 520.0, n), index=idx, dtype=float)
    ig_oas = pd.Series(np.linspace(110.0, 180.0, n), index=idx, dtype=float)
    realized_vol = pd.Series(np.linspace(0.2, 0.95, n), index=idx, dtype=float)
    avg_corr = pd.Series(np.linspace(0.1, 0.9, n), index=idx, dtype=float)

    feats = compute_credit_funding_features(
        hyg_close=hyg,
        lqd_close=lqd,
        tlt_close=tlt,
        kre_close=kre,
        spy_close=spy,
        sofr=sofr,
        iorb=iorb,
        nfci_weekly=nfci,
        broad_usd_index=usd,
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=_default_rules(),
    )
    precomputed = build_rule_inputs_by_date(
        features=feats,
        hy_spread_percentile_504d=feats.hy_oas_percentile_504d,
        hy_spread_slope_21d=feats.hy_oas_slope_21d,
        ig_spread_slope_21d=feats.ig_oas_slope_21d,
        realized_vol_21d_percentile_252d=realized_vol,
        avg_pairwise_corr_percentile_504d=avg_corr,
    )
    for dt in idx[504::37]:
        expected = build_rule_inputs_for_date(
            features=feats,
            dt=dt,
            hy_spread_percentile_504d=feats.hy_oas_percentile_504d,
            hy_spread_slope_21d=feats.hy_oas_slope_21d,
            ig_spread_slope_21d=feats.ig_oas_slope_21d,
            realized_vol_21d_percentile_252d=realized_vol,
            avg_pairwise_corr_percentile_504d=avg_corr,
        )
        actual = precomputed[dt]
        for field in expected.__dataclass_fields__:
            assert getattr(actual, field) == pytest.approx(
                getattr(expected, field), nan_ok=True
            )


def test_build_rule_inputs_accepts_either_spread_source() -> None:
    """The same builder, pointed at the OAS triple vs the proxy triple,
    yields rule inputs whose source-neutral spread fields differ while the
    shared macro/vol fields match (Ambiguity Log #71)."""
    idx = _bdate_index(periods=600)
    common = dict(
        hyg_close=_make_random_walk(idx, seed=21, start=80.0, sigma=0.004),
        lqd_close=_make_random_walk(idx, seed=22, start=110.0, sigma=0.003),
        tlt_close=_make_random_walk(idx, seed=23, start=95.0, sigma=0.005),
        kre_close=_make_random_walk(idx, seed=24, start=55.0, sigma=0.006),
        spy_close=_make_random_walk(idx, seed=25, start=420.0, sigma=0.008),
        sofr=_make_constant_series(idx, 5.3, "sofr"),
        iorb=_make_constant_series(idx, 5.4, "iorb"),
        nfci_weekly=_make_constant_series(idx, -0.2, "nfci"),
        broad_usd_index=_make_random_walk(idx, seed=26, start=120.0, sigma=0.003),
        hy_oas=_make_random_walk(idx, seed=27, start=3.5, sigma=0.05),
        ig_oas=_make_random_walk(idx, seed=28, start=1.2, sigma=0.03),
        config=_default_rules(),
    )
    f = compute_credit_funding_features(**common)
    rvp = _make_constant_series(idx, 0.5, "rvp")
    acp = _make_constant_series(idx, 0.5, "acp")
    dt = idx[550]

    oas_inputs = build_rule_inputs_for_date(
        features=f,
        dt=dt,
        hy_spread_percentile_504d=f.hy_oas_percentile_504d,
        hy_spread_slope_21d=f.hy_oas_slope_21d,
        ig_spread_slope_21d=f.ig_oas_slope_21d,
        realized_vol_21d_percentile_252d=rvp,
        avg_pairwise_corr_percentile_504d=acp,
    )
    proxy_inputs = build_rule_inputs_for_date(
        features=f,
        dt=dt,
        hy_spread_percentile_504d=f.hy_tr_differential_percentile_504d,
        hy_spread_slope_21d=f.hy_tr_differential_slope_21d,
        ig_spread_slope_21d=f.ig_tr_differential_slope_21d,
        realized_vol_21d_percentile_252d=rvp,
        avg_pairwise_corr_percentile_504d=acp,
    )
    # Source-neutral spread fields differ between the two metrics...
    assert (
        oas_inputs.hy_spread_percentile_504d != proxy_inputs.hy_spread_percentile_504d
    )
    # ...while the shared macro/vol fields are identical.
    assert oas_inputs.spy_21d_return == proxy_inputs.spy_21d_return
    assert oas_inputs.sofr_iorb_slope_21d == proxy_inputs.sofr_iorb_slope_21d


# --- Group B — Rule precedence (6 tests) -------------------------------------


def _rule_inputs(**overrides: float) -> CreditFundingRuleInputs:
    """Build a CreditFundingRuleInputs with neutral defaults; override per-test."""
    defaults: dict[str, float] = dict(
        hy_spread_percentile_504d=0.50,
        hy_spread_slope_21d=0.0,
        ig_spread_slope_21d=0.0,
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
        hy_spread_percentile_504d=0.30,
        hy_spread_slope_21d=-0.001,
    )
    assert evaluate_credit_calm(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "credit_calm"


def test_spread_widening_fires_on_strict_positive_slopes_both_legs() -> None:
    """§2C lines 2069-2071: both HY and IG slopes strictly > 0."""
    rules = _default_rules()
    inputs = _rule_inputs(
        hy_spread_percentile_504d=0.50,
        hy_spread_slope_21d=0.002,
        ig_spread_slope_21d=0.001,
    )
    assert evaluate_spread_widening(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "spread_widening"


def test_credit_stress_fires_on_high_percentile_and_falling_spy() -> None:
    """§2C lines 2073-2075: pct=0.85 AND spy_21d=-0.06."""
    rules = _default_rules()
    inputs = _rule_inputs(
        hy_spread_percentile_504d=0.85,
        spy_21d_return=-0.06,
    )
    assert evaluate_credit_stress(inputs, rules) is True
    assert evaluate_rules(inputs=inputs, config=rules) == "credit_stress"


def test_funding_squeeze_fires_on_usd_zscore_and_sofr_widening_and_falling_spy() -> (
    None
):
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
        hy_spread_percentile_504d=0.85,
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
    omit_oas_series: bool = False,
):
    """Build a MarketContext with full cross_asset_closes and macro_series.

    Optional knobs simulate spec unknown-gate failure modes.
    """
    idx = _bdate_index(periods=_TRAINING_SESSIONS)
    n = len(idx)
    rng = np.random.default_rng(_SEED)

    # Build full NETWORK_FRAGILITY_UNIVERSE prices (so feature_store.network_fragility lights up).
    universe_prices = pd.DataFrame(
        (
            1.0 + rng.normal(0.0, 0.01, size=(n, len(NETWORK_FRAGILITY_UNIVERSE)))
        ).cumprod(axis=0)
        * 100.0,
        index=idx,
        columns=list(NETWORK_FRAGILITY_UNIVERSE),
    )
    spy_close = universe_prices[INDEX_SYMBOL]
    market_rows: list[dict[str, object]] = []
    for ts in idx:
        close = float(spy_close.loc[ts])
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "SPY",
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }
        )
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "RSP",
                "open": close * 0.5,
                "high": close * 0.5 * 1.005,
                "low": close * 0.5 * 0.995,
                "close": close * 0.5,
                "volume": 500_000,
            }
        )
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "VIXY",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100_000,
            }
        )
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
        "SOFR": sofr,
        "IORB": iorb,
        "NFCI": nfci_w,
        "broad_usd_index": usd,
        # ICE BofA OAS series — single source for the §2C credit-spread
        # metric. Required by `_CF_MACRO_KEYS`, so the §2C seam does not
        # build without them.
        "hy_oas": _make_random_walk(idx, seed=_SEED + 101, start=400.0, sigma=0.01),
        "ig_bbb_oas": _make_random_walk(idx, seed=_SEED + 102, start=150.0, sigma=0.01),
        # Add yield series for monetary slice compatibility.
        "DGS2": _make_constant_series(idx, 4.5, "DGS2"),
        "DGS10": _make_constant_series(idx, 4.0, "DGS10"),
    }
    if omit_oas_series:
        macro_series.pop("hy_oas")
        macro_series.pop("ig_bbb_oas")

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
    return store, build_credit_funding_axis_series(context, store)


def _build_real_v2_credit_context(
    as_of: date,
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
):
    required_symbols = set(SECTOR_ETFS) | set(CROSS_ASSET_SYMBOLS) | {"KRE"}
    missing = required_symbols - set(v2_close_series_by_symbol)
    assert not missing, f"V2 OHLCV fixture missing symbols: {sorted(missing)}"
    sector_etf_closes = {
        symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
    }
    cross_asset_closes = {
        symbol: v2_close_series_by_symbol[symbol]
        for symbol in set(CROSS_ASSET_SYMBOLS) | {"KRE"}
    }
    return build_market_context(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        config=RegimeEngine().config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=v2_macro_series_by_key,
    )


def test_build_proxy_runs_parallel_to_build_with_proxy_bias_code() -> None:
    """build_proxy() runs the identical §2C rule schema on the TLT-proxy
    series, producing a parallel output keyed exactly like build() — but
    tagged with the proxy bias-warning code, never blended (Log #71)."""
    context = _build_full_synthetic_context()
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    real = build_credit_funding_axis_series(context, store)
    proxy = build_credit_funding_proxy_axis_series(context, store)

    assert real is not None and proxy is not None
    # One output per session from both runs.
    assert set(real.keys()) == set(proxy.keys())

    # A session where the rule engine fired (past every cold-start / gate) in
    # both runs — its evidence must carry the source-specific bias code.
    rule_day = next(
        d
        for d in real
        if "rule_evidence" in real[d].evidence and "rule_evidence" in proxy[d].evidence
    )
    assert (
        real[rule_day].evidence["bias_warning_code"]
        == "credit_spread_ice_bofa_oas_fred"
    )
    assert (
        proxy[rule_day].evidence["bias_warning_code"]
        == "credit_spread_proxy_total_return_differential"
    )
    assert real[rule_day].evidence["spread_source"] == "ice_bofa_oas"
    assert proxy[rule_day].evidence["spread_source"] == "tlt_total_return_differential"


def _credit_output(
    *,
    label: str,
    source: str,
    status: str = "ok",
) -> CreditFundingOutput:
    return CreditFundingOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={"spread_source": source},
        data_quality=DataQuality(status=status),
    )


def test_effective_credit_funding_uses_higher_risk_when_oas_and_proxy_diverge() -> None:
    oas = _credit_output(label="credit_calm", source="ice_bofa_oas")
    proxy = _credit_output(
        label="spread_widening",
        source="tlt_total_return_differential",
    )

    effective = resolve_credit_funding_effective_output(oas=oas, proxy=proxy)

    assert effective is not None
    assert effective.active_label == "spread_widening"
    assert effective.evidence["source_used"] == "proxy_higher_risk"
    assert effective.evidence["agreement_status"] == "divergent"
    assert effective.evidence["oas_label"] == "credit_calm"
    assert effective.evidence["proxy_label"] == "spread_widening"


def test_effective_credit_funding_falls_back_to_proxy_when_oas_unavailable() -> None:
    oas = _credit_output(
        label="unknown",
        source="ice_bofa_oas",
        status="insufficient_data",
    )
    proxy = _credit_output(
        label="credit_calm",
        source="tlt_total_return_differential",
    )

    effective = resolve_credit_funding_effective_output(oas=oas, proxy=proxy)

    assert effective is not None
    assert effective.active_label == "credit_calm"
    assert effective.evidence["source_used"] == "proxy_fallback"
    assert effective.evidence["agreement_status"] == "proxy_only"


def test_credit_funding_proxy_builds_when_oas_series_are_absent() -> None:
    context = _build_full_synthetic_context(omit_oas_series=True)
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    real = build_credit_funding_axis_series(context, store)
    proxy = build_credit_funding_proxy_axis_series(context, store)

    assert real is not None
    assert proxy is not None
    rule_day = next(
        d
        for d in proxy
        if "rule_evidence" in proxy[d].evidence and proxy[d].active_label != "unknown"
    )
    assert real[rule_day].active_label == "unknown"
    assert proxy[rule_day].active_label in CREDIT_FUNDING_RISK_RANK
    effective = resolve_credit_funding_effective_output(
        oas=real[rule_day],
        proxy=proxy[rule_day],
    )
    assert effective is not None
    assert effective.evidence["source_used"] == "proxy_fallback"


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


def test_credit_funding_carries_one_session_sofr_publication_lag() -> None:
    """SOFR can be absent on the latest NYSE session until publication catches up."""
    context = _build_full_synthetic_context(sofr_drop_last=True)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label != "unknown"
    assert out.data_quality.status != "insufficient_data"


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
    nan_series = pd.Series(np.nan, index=cf.hy_oas_63d.index)
    broken = CreditFundingFeatures(
        hy_oas_63d=nan_series,
        ig_oas_63d=nan_series,
        hy_oas_percentile_504d=nan_series,
        hy_oas_slope_21d=nan_series,
        ig_oas_slope_21d=nan_series,
        hy_tr_differential_63d=nan_series,
        ig_tr_differential_63d=nan_series,
        hy_tr_differential_percentile_504d=nan_series,
        hy_tr_differential_slope_21d=nan_series,
        ig_tr_differential_slope_21d=nan_series,
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
    outputs = build_credit_funding_axis_series(context, broken_store)
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


def test_feature_store_credit_funding_seam_none_without_kre_in_cross_asset_closes() -> (
    None
):
    """Missing KRE on cross_asset_closes → feature_store.credit_funding is None."""
    context = _build_full_synthetic_context()
    # Strip KRE from the cross_asset_closes dict.
    stripped = {
        k: v for k, v in (context.cross_asset_closes or {}).items() if k != "KRE"
    }
    new_context = build_market_context(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(),
                    "symbol": "SPY",
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
                    "date": ts.date(),
                    "symbol": "RSP",
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


def test_real_v2_fixture_credit_funding_golden_label(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
) -> None:
    """Real V2 OHLCV + FRED fixture lights §2C and pins current labels."""
    as_of = _REAL_FIXTURE_CREDIT_AS_OF
    context = _build_real_v2_credit_context(
        as_of,
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
        v2_macro_series_by_key,
    )
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        credit_funding_config=context.config.credit_funding,
    )
    assert store.credit_funding is not None

    real_outputs = build_credit_funding_axis_series(context, store)
    proxy_outputs = build_credit_funding_proxy_axis_series(context, store)
    assert real_outputs is not None
    assert proxy_outputs is not None

    real = real_outputs[as_of]
    proxy = proxy_outputs[as_of]
    assert real.raw_label == "credit_calm"
    assert real.stable_label == "credit_calm"
    assert real.active_label == "credit_calm"
    assert real.data_quality.status == "ok"
    assert real.data_quality.reason is None
    assert real.evidence["spread_source"] == "ice_bofa_oas"
    assert real.evidence["bias_warning_code"] == "credit_spread_ice_bofa_oas_fred"
    assert real.evidence["nfci_daily_carried"] == pytest.approx(-0.524)
    assert real.evidence["kre_spy_slope_63d"] == pytest.approx(-7.786519147306989e-05)
    real_rule = real.evidence["rule_evidence"]
    assert real_rule["hy_spread_percentile_504d"] == pytest.approx(0.24305555555555555)
    assert real_rule["hy_spread_slope_21d"] == pytest.approx(-0.004064935064935065)
    assert real_rule["spy_21d_return"] == pytest.approx(0.07590730214254471)
    assert real_rule["avg_pairwise_corr_percentile_504d"] == pytest.approx(
        0.3055555555555556
    )

    assert proxy.raw_label == "credit_calm"
    assert proxy.stable_label == "credit_calm"
    assert proxy.active_label == "credit_calm"
    assert proxy.data_quality.status == "ok"
    assert proxy.evidence["spread_source"] == "tlt_total_return_differential"
    assert (
        proxy.evidence["bias_warning_code"]
        == "credit_spread_proxy_total_return_differential"
    )
    proxy_rule = proxy.evidence["rule_evidence"]
    assert proxy_rule["hy_spread_percentile_504d"] == pytest.approx(0.31746031746031744)
    assert proxy_rule["hy_spread_slope_21d"] == pytest.approx(-0.0005359273106014766)


def test_regime_output_carries_real_fixture_credit_funding_state_when_configured(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
) -> None:
    """End-to-end: real fixture reaches both §2C wire fields."""
    as_of = _REAL_FIXTURE_CREDIT_AS_OF
    context = _build_real_v2_credit_context(
        as_of,
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
        v2_macro_series_by_key,
    )
    engine = RegimeEngine()
    timeline = engine.classify_window(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        lookback_days=1,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=context.cross_asset_closes,
        macro_series=v2_macro_series_by_key,
    )
    out = timeline.outputs[-1]
    assert out.as_of_date == as_of
    assert out.credit_funding_state is not None
    assert out.credit_funding_state.raw_label == "credit_calm"
    assert out.credit_funding_state.active_label == "credit_calm"
    assert out.credit_funding_state.evidence["spread_source"] == "ice_bofa_oas"
    assert out.credit_funding_state_proxy is not None
    assert out.credit_funding_state_proxy.raw_label == "credit_calm"
    assert out.credit_funding_state_proxy.active_label == "credit_calm"
    assert (
        out.credit_funding_state_proxy.evidence["spread_source"]
        == "tlt_total_return_differential"
    )
    assert out.credit_funding_effective_state is not None
    assert out.credit_funding_effective_state.active_label == "credit_calm"
    assert out.credit_funding_effective_state.evidence["agreement_status"] in {
        "confirmed",
        "divergent",
    }


def test_regime_output_carries_credit_funding_state_when_configured() -> None:
    """End-to-end: classify_window populates RegimeOutput.credit_funding_state."""
    context = _build_full_synthetic_context()
    engine = RegimeEngine()
    timeline = engine.classify_window(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(),
                    "symbol": "SPY",
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
                    "date": ts.date(),
                    "symbol": "RSP",
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
    # §2C parallel proxy label (Ambiguity Log #71) — emitted alongside the
    # real-OAS label, a distinct CreditFundingOutput, never blended.
    assert out.credit_funding_state_proxy is not None
    assert out.credit_funding_state_proxy is not out.credit_funding_state
    assert out.credit_funding_state_proxy.active_label in allowed
    assert out.credit_funding_effective_state is not None
    assert out.credit_funding_effective_state.active_label in allowed
