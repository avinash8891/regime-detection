"""TDD tests for v2 §3.2 Network Fragility feature compute.

Per ~/.claude/CLAUDE.md: realistic v2 production symbols (XLB, XLK, SPY, TLT
etc) — NO toy a/b/c names. Math is verified against numpy/pandas baselines
or hand-computed eigen-decompositions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    INDEX_SYMBOL,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)
from regime_detection.network_fragility import (
    NetworkFragilityFeatures,
    compute_features,
)


# ---------- Shared fixtures ----------------------------------------------------


@pytest.fixture
def nyse_index_100() -> pd.DatetimeIndex:
    """100 business days ending 2024-12-31."""
    return pd.bdate_range(end="2024-12-31", periods=100)


@pytest.fixture
def small_4asset_returns(nyse_index_100) -> pd.DataFrame:
    """4 real v2 symbols (XLB, XLK, SPY, TLT), 100 business days."""
    rng = np.random.default_rng(seed=20260512)
    cols = ["XLB", "XLK", "SPY", "TLT"]
    data = rng.normal(loc=0.0, scale=0.01, size=(100, 4))
    return pd.DataFrame(data, index=nyse_index_100, columns=cols)


def _prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    return (1.0 + returns.fillna(0.0)).cumprod() * start


def _split_to_kwargs(prices: pd.DataFrame) -> dict[str, object]:
    """Split a 4-col real-symbol price frame into compute_features kwargs."""
    spy_close = prices["SPY"]
    sector = {c: prices[c] for c in prices.columns if c in SECTOR_ETFS}
    cross = {c: prices[c] for c in prices.columns if c in CROSS_ASSET_SYMBOLS}
    return {
        "sector_etf_closes": sector,
        "cross_asset_closes": cross,
        "spy_close": spy_close,
    }


def _make_full_universe_prices(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    rets = rng.normal(0.0, 0.01, size=(len(index), len(NETWORK_FRAGILITY_UNIVERSE)))
    df = pd.DataFrame(rets, index=index, columns=list(NETWORK_FRAGILITY_UNIVERSE))
    return _prices_from_returns(df)


def _full_universe_kwargs(prices: pd.DataFrame) -> dict[str, object]:
    return {
        "sector_etf_closes": {s: prices[s] for s in SECTOR_ETFS},
        "cross_asset_closes": {s: prices[s] for s in CROSS_ASSET_SYMBOLS},
        "spy_close": prices[INDEX_SYMBOL],
    }


# ---------- Feature unit tests -------------------------------------------------


def test_avg_pairwise_corr_63d_against_pandas_corr_baseline(small_4asset_returns):
    """Avg upper-triangle off-diagonal correlation at session 70 must match
    pandas .corr() over the same 63d window."""
    prices = _prices_from_returns(small_4asset_returns)
    out = compute_features(
        **_split_to_kwargs(prices),
        min_universe_size=4,
    )

    target_dt = prices.index[70]
    rets = prices.pct_change(fill_method=None).loc[:target_dt].tail(63)
    corr = rets.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    expected = corr[iu].mean()

    assert out.avg_pairwise_corr_63d.loc[target_dt] == pytest.approx(expected, abs=1e-12)


def test_largest_eigenvalue_share_against_numpy_linalg(nyse_index_100):
    """Build a 1-factor returns frame with high pairwise correlation;
    the first eigenvalue should dominate, and the function's
    largest_eigenvalue_share at the final session must match the
    direct numpy decomposition of the same 63d correlation matrix."""
    rng = np.random.default_rng(7)
    factor = rng.normal(0.0, 0.01, size=len(nyse_index_100))
    noise = rng.normal(0.0, 0.001, size=(len(nyse_index_100), 4))
    cols = ["XLB", "XLK", "SPY", "TLT"]
    rets = pd.DataFrame(
        factor[:, None] + noise,
        index=nyse_index_100,
        columns=cols,
    )
    prices = _prices_from_returns(rets)
    out = compute_features(**_split_to_kwargs(prices), min_universe_size=4)

    target_dt = prices.index[-1]
    window = prices.pct_change(fill_method=None).loc[:target_dt].tail(63)
    corr = np.corrcoef(window.to_numpy(), rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    expected_share = eigs[0] / eigs.sum()

    assert out.largest_eigenvalue_share.loc[target_dt] == pytest.approx(
        expected_share, abs=1e-10
    )
    # Sanity: highly factor-driven → first eigenvalue near N=4.
    assert out.largest_eigenvalue_share.loc[target_dt] > 0.95


def test_effective_rank_for_identity_matrix_equals_n(nyse_index_100):
    """4 truly independent assets → correlation ≈ I → eigenvalues all ≈ 1
    → effective rank ≈ 4."""
    rng = np.random.default_rng(2024)
    # Long sample minimises sampling correlation so corr matrix → identity.
    long_index = pd.bdate_range(end="2024-12-31", periods=2000)
    cols = ["XLB", "XLK", "SPY", "TLT"]
    rets = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(long_index), 4)),
        index=long_index,
        columns=cols,
    )
    prices = _prices_from_returns(rets)

    # Use a long correlation lookback so sample noise is small.
    out = compute_features(
        **_split_to_kwargs(prices),
        correlation_lookback_days=1500,
        min_universe_size=4,
    )
    target_dt = prices.index[-1]
    assert out.effective_rank.loc[target_dt] == pytest.approx(4.0, abs=0.1)


def test_effective_rank_for_perfect_correlation_equals_one(nyse_index_100):
    """All assets identical → rank-1 corr → effective rank = 1."""
    rng = np.random.default_rng(11)
    factor = rng.normal(0.0, 0.01, size=len(nyse_index_100))
    cols = ["XLB", "XLK", "SPY", "TLT"]
    rets = pd.DataFrame(
        np.tile(factor[:, None], (1, 4)),
        index=nyse_index_100,
        columns=cols,
    )
    prices = _prices_from_returns(rets)
    out = compute_features(**_split_to_kwargs(prices), min_universe_size=4)

    target_dt = prices.index[-1]
    assert out.effective_rank.loc[target_dt] == pytest.approx(1.0, abs=1e-6)


def test_absorption_ratio_top3_against_hand_computed_eigenvalues(small_4asset_returns):
    """At final session, absorption_ratio_top3 should equal
    sum(top-3 eigenvalues) / sum(all eigenvalues) of the 63d corr matrix."""
    prices = _prices_from_returns(small_4asset_returns)
    out = compute_features(**_split_to_kwargs(prices), min_universe_size=4)

    target_dt = prices.index[-1]
    window = prices.pct_change(fill_method=None).loc[:target_dt].tail(63)
    corr = np.corrcoef(window.to_numpy(), rowvar=False)
    eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    expected = eigs[:3].sum() / eigs.sum()

    assert out.absorption_ratio_top3.loc[target_dt] == pytest.approx(expected, abs=1e-12)


def test_dispersion_ratio_against_hand_computed_realized_vols(small_4asset_returns):
    """dispersion_ratio = mean(21d realized vol across symbols) / SPY 21d vol."""
    prices = _prices_from_returns(small_4asset_returns)
    out = compute_features(**_split_to_kwargs(prices), min_universe_size=4)

    target_dt = prices.index[-1]
    rets = prices.pct_change(fill_method=None)
    window = rets.loc[:target_dt].tail(21)
    per_symbol_vol = window.std(ddof=1) * np.sqrt(252)
    mean_vol = per_symbol_vol.mean()
    spy_vol = per_symbol_vol["SPY"]
    expected = mean_vol / spy_vol

    assert out.dispersion_ratio.loc[target_dt] == pytest.approx(expected, abs=1e-12)


def test_compute_features_emits_nan_when_universe_below_min_size(nyse_index_100):
    """If only 19 of 22 symbols survive the completeness filter,
    all features must be NaN (data quality layer flags unknown downstream)."""
    # Build full universe but corrupt 3 symbols with all-NaN in the window.
    prices = _make_full_universe_prices(nyse_index_100, seed=42)
    drop_symbols = ["UUP", "USO", "LQD"]
    prices.loc[prices.index[-63:], drop_symbols] = np.nan

    out = compute_features(**_full_universe_kwargs(prices))
    target_dt = prices.index[-1]

    assert pd.isna(out.avg_pairwise_corr_63d.loc[target_dt])
    assert pd.isna(out.largest_eigenvalue_share.loc[target_dt])
    assert pd.isna(out.effective_rank.loc[target_dt])
    assert pd.isna(out.absorption_ratio_top3.loc[target_dt])


def test_compute_features_drops_columns_below_completeness_threshold(nyse_index_100):
    """Inject 50% NaNs into XLE's window — it should be dropped before
    correlation computation, leaving 21 surviving symbols, and the result
    must equal the corr computed without XLE."""
    prices = _make_full_universe_prices(nyse_index_100, seed=99)
    # Corrupt half of XLE's last 63 sessions.
    xle_window_idx = prices.index[-63:]
    corrupt_idx = xle_window_idx[::2]  # 50% NaN
    prices.loc[corrupt_idx, "XLE"] = np.nan

    out = compute_features(**_full_universe_kwargs(prices))

    target_dt = prices.index[-1]
    rets = prices.pct_change(fill_method=None)
    window = rets.loc[:target_dt].tail(63)
    surviving = [c for c in window.columns if c != "XLE"]
    sub = window[surviving].dropna(axis=0, how="any")
    corr = np.corrcoef(sub.to_numpy(), rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    expected = corr[iu].mean()

    assert out.avg_pairwise_corr_63d.loc[target_dt] == pytest.approx(expected, abs=1e-10)


def test_compute_features_percentile_504d_against_rolling_rank(small_4asset_returns):
    """The percentile column must equal pandas rolling(window).rank(pct=True)
    applied to the raw feature series."""
    prices = _prices_from_returns(small_4asset_returns)
    out = compute_features(
        **_split_to_kwargs(prices),
        min_universe_size=4,
        percentile_lookback_days=30,  # short window so we get values in 100-day fixture
    )
    expected = out.avg_pairwise_corr_63d.rolling(30).rank(pct=True)
    target_dt = prices.index[-1]
    assert out.avg_pairwise_corr_percentile_504d.loc[target_dt] == pytest.approx(
        expected.loc[target_dt], abs=1e-12
    )


def test_compute_features_returns_series_aligned_to_spy_index(small_4asset_returns):
    """All output series must share the SPY DatetimeIndex exactly."""
    prices = _prices_from_returns(small_4asset_returns)
    out = compute_features(**_split_to_kwargs(prices), min_universe_size=4)
    spy_index = prices["SPY"].index

    assert isinstance(out, NetworkFragilityFeatures)
    for name in [
        "avg_pairwise_corr_63d",
        "avg_pairwise_corr_percentile_504d",
        "largest_eigenvalue_share",
        "largest_eigenvalue_share_percentile_504d",
        "effective_rank",
        "effective_rank_percentile_504d",
        "absorption_ratio_top3",
        "dispersion_ratio",
        "dispersion_ratio_percentile_252d",
    ]:
        s = getattr(out, name)
        assert isinstance(s, pd.Series), name
        assert s.index.equals(spy_index), name
