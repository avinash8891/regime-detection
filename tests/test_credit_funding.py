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

from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    CreditFundingRulesConfig,
    load_default_regime_config,
)
from regime_detection.credit_funding import (
    CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
    CREDIT_SPREAD_SOURCE,
    CREDIT_SPREAD_SOURCE_CODE,
    CREDIT_SPREAD_SOURCE_URL,
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
    sofr = _make_constant_series(idx, 5.0, "sofr")
    iorb = _make_constant_series(idx, 4.9, "iorb")
    # NFCI weekly: assign every 5th index; rest NaN so ffill exercises.
    nfci_w = pd.Series(np.nan, index=idx, name="nfci", dtype=float)
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


# --- Group G — Pre-SOFR/IORB splice regression (ADR 0009) --------------------
#
# Regression guard: this bug has been resolved multiple times. The splice in
# compute_credit_funding_features (credit_funding.py:409-416) fills the
# sofr_iorb_spread for pre-SOFR (pre-2018-04-03) and pre-IORB (pre-2021-07-29)
# sessions using FEDFUNDS-IOER. If fedfunds/ioer_legacy are not passed (or the
# feature_store stops routing them), sofr_iorb_spread is all-NaN for 2016-2021
# and the axis builder emits stale_data for 67% of history.


def _pre_sofr_index() -> pd.DatetimeIndex:
    """650 business days ending 2017-12-29 — entirely before SOFR (Apr 2018)."""
    return pd.bdate_range(end="2017-12-29", periods=650)


def _make_pre_sofr_features(*, fedfunds: pd.Series | None, ioer_legacy: pd.Series | None):
    """Build CreditFundingFeatures over a pre-SOFR index with the splice inputs."""
    idx = _pre_sofr_index()
    rng = np.random.default_rng(_SEED)
    rw = lambda start, sigma: pd.Series(  # noqa: E731
        start * (1.0 + rng.normal(0.0, sigma, size=len(idx))).cumprod(),
        index=idx,
        dtype=float,
    )
    # SOFR and IORB are both NaN for this pre-2018 window (they don't exist yet).
    nan_series = pd.Series(float("nan"), index=idx, dtype=float)
    nfci_w = pd.Series(float("nan"), index=idx, dtype=float)
    nfci_w.iloc[::5] = rng.normal(-0.5, 0.2, size=len(nfci_w.iloc[::5]))
    return compute_credit_funding_features(
        hyg_close=rw(80.0, 0.008),
        lqd_close=rw(110.0, 0.006),
        tlt_close=rw(100.0, 0.006),
        kre_close=rw(50.0, 0.012),
        spy_close=rw(2000.0, 0.008),
        sofr=nan_series,
        iorb=nan_series,
        nfci_weekly=nfci_w,
        broad_usd_index=rw(100.0, 0.003),
        hy_oas=rw(400.0, 0.01),
        ig_oas=rw(150.0, 0.01),
        config=_default_rules(),
        fedfunds=fedfunds,
        ioer_legacy=ioer_legacy,
    )


def test_sofr_iorb_splice_fills_pre_sofr_era_when_fedfunds_ioer_supplied() -> None:
    """ADR 0009 splice regression: with fedfunds+ioer_legacy, sofr_iorb_spread
    is 100% non-null for 2016-2017 (pre-SOFR, pre-IORB).

    Without this, the axis builder emits stale_data for 67% of full history.
    """
    idx = _pre_sofr_index()
    fedfunds = pd.Series(0.41, index=idx, dtype=float, name="fedfunds")  # ~2017 FEDFUNDS
    ioer_legacy = pd.Series(0.40, index=idx, dtype=float, name="ioer_legacy")

    features = _make_pre_sofr_features(fedfunds=fedfunds, ioer_legacy=ioer_legacy)

    null_count = features.sofr_iorb_spread.isna().sum()
    assert null_count == 0, (
        f"sofr_iorb_spread has {null_count} NaN values in pre-SOFR era — "
        "fedfunds/ioer_legacy splice is broken. Check feature_store.py:616-617 "
        "and credit_funding.py:409-416."
    )


def test_sofr_iorb_spread_all_nan_without_splice_inputs() -> None:
    """Confirm that WITHOUT fedfunds/ioer_legacy, sofr_iorb_spread is all-NaN
    for a pre-SOFR index. This documents the broken state the splice fixes,
    so that the above test's assertion is meaningful and not trivially true.
    """
    features = _make_pre_sofr_features(fedfunds=None, ioer_legacy=None)

    non_null_count = features.sofr_iorb_spread.notna().sum()
    assert non_null_count == 0, (
        f"Expected all-NaN without splice, got {non_null_count} non-null — "
        "the pre-SOFR/IORB window baseline assumption has changed."
    )


# --- Group C — Unknown gate (4 tests) ----------------------------------------
