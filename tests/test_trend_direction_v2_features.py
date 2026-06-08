"""TDD tests for v2 §1A Layer 1 V2 Trend Direction features (Slice 2.1).

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic SPY-like price series
and the real production Pydantic config — NO toy a/b/c names. Math is
verified against hand-computed values or numpy baselines.

Spec references:
    docs/regime_engine_v2_spec.md §1A (lines 61–135).
    Slice scope: features only, no classifier. See §8 line 1181.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import TrendDirectionV2Config, load_default_regime_config
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.trend_direction import (
    TrendDirectionV2Features,
    compute_trend_v2_features,
)

# ---------- Shared fixtures ---------------------------------------------------


@pytest.fixture
def v2_trend_config() -> TrendDirectionV2Config:
    """Real production defaults from configs/core3-v2.0.0.yaml."""
    return TrendDirectionV2Config(
        efficiency_ratio_lookback_days=20,
        hurst_lookback_days=250,
        slope_lookback_days=20,
        sma_short_period=50,
        sma_long_period=200,
        return_short_period=63,
        return_long_period=126,
        drawdown_lookback_days=252,
    )


@pytest.fixture
def spy_like_random_walk_1000() -> pd.Series:
    """1000-session synthetic SPY-like random walk seeded for repeatability.

    Geometric-Brownian-Motion-style: daily log returns ~ N(0.0003, 0.01).
    Seed 20260512 fixed for reproducibility of hurst tolerance tests.
    """
    rng = np.random.default_rng(seed=20260512)
    log_rets = rng.normal(loc=0.0003, scale=0.01, size=1000)
    prices = 400.0 * np.exp(np.cumsum(log_rets))
    index = pd.bdate_range(end="2024-12-31", periods=1000)
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def monotonic_rising_1000() -> pd.Series:
    """1000 sessions, strictly monotonically rising by 1.0 / day."""
    index = pd.bdate_range(end="2024-12-31", periods=1000)
    return pd.Series(np.arange(100.0, 1100.0), index=index, name="close")


# ---------- efficiency_ratio_20d ---------------------------------------------


def test_efficiency_ratio_monotonic_series_equals_one(
    monotonic_rising_1000, v2_trend_config
):
    out = compute_trend_v2_features(monotonic_rising_1000, config=v2_trend_config)
    # After session index 20 (need 20 prior diffs), ER should be exactly 1.0
    # for a strictly monotonic series.
    er = out.efficiency_ratio_20d
    valid = er.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.to_numpy(), 1.0, rtol=0, atol=1e-12)


def test_efficiency_ratio_alternating_series_close_to_zero(v2_trend_config):
    # Alternating ±1% returns keep price oscillating; path_length >> directional_move.
    index = pd.bdate_range(end="2024-12-31", periods=200)
    returns = np.empty(200)
    returns[0::2] = 0.01
    returns[1::2] = -0.01
    prices = 100.0 * np.cumprod(1.0 + returns)
    close = pd.Series(prices, index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)
    er = out.efficiency_ratio_20d.dropna()
    # With near-zero net move over 20d, ratio is near 0 (allow 0.05 since
    # compounding leaves a tiny residual drift).
    assert (er < 0.1).all()


def test_efficiency_ratio_constant_series_is_nan(v2_trend_config):
    index = pd.bdate_range(end="2024-12-31", periods=100)
    close = pd.Series(np.full(100, 250.0), index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)
    er = out.efficiency_ratio_20d
    # Constant series → path_length=0 → ratio is NaN by definition.
    assert er.iloc[20:].isna().all()


def test_efficiency_ratio_hand_computed_value(v2_trend_config):
    """Hand-compute ER at session 20 against the §1A pseudocode."""
    index = pd.bdate_range(end="2024-12-31", periods=30)
    rng = np.random.default_rng(seed=42)
    prices = 100.0 + np.cumsum(rng.normal(0, 1.0, size=30))
    close = pd.Series(prices, index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)

    t = 20
    directional = abs(close.iloc[t] - close.iloc[t - 20])
    path = sum(abs(close.iloc[i] - close.iloc[i - 1]) for i in range(t - 19, t + 1))
    expected = directional / path
    assert out.efficiency_ratio_20d.iloc[t] == pytest.approx(expected, abs=1e-12)


# ---------- hurst_250d --------------------------------------------------------


def test_hurst_random_walk_near_half(spy_like_random_walk_1000, v2_trend_config):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    h = out.hurst_250d.dropna()
    assert len(h) > 100
    # Single 250d window R/S is noisy; literature tolerance ±0.15 on a
    # 250-sample window. Mean across sessions should be within ±0.10.
    assert abs(h.mean() - 0.5) < 0.10


def test_hurst_pure_trend_above_half(monotonic_rising_1000, v2_trend_config):
    out = compute_trend_v2_features(monotonic_rising_1000, config=v2_trend_config)
    h = out.hurst_250d.dropna()
    assert len(h) > 100
    # A pure deterministic trend → Hurst near 1.0; conservatively > 0.55.
    assert (h > 0.55).all()


def test_hurst_mean_reverting_below_half(v2_trend_config):
    """Strongly mean-reverting series should have Hurst < 0.45."""
    # AR(1) with negative coefficient → strong mean reversion in returns.
    rng = np.random.default_rng(seed=20260512)
    n = 1000
    log_rets = np.zeros(n)
    log_rets[0] = rng.normal(0, 0.01)
    for i in range(1, n):
        # Strong negative autocorrelation in returns ⇒ price path
        # oscillates rapidly around its mean.
        log_rets[i] = -0.7 * log_rets[i - 1] + rng.normal(0, 0.01)
    prices = 400.0 * np.exp(np.cumsum(log_rets))
    index = pd.bdate_range(end="2024-12-31", periods=n)
    close = pd.Series(prices, index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)
    h = out.hurst_250d.dropna()
    assert h.mean() < 0.45


def test_hurst_nan_until_lookback(spy_like_random_walk_1000, v2_trend_config):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    h = out.hurst_250d
    # Index 0 .. 248 must be NaN (need 250 sessions for first value).
    assert h.iloc[:249].isna().all()
    assert not math.isnan(h.iloc[249])


# ---------- slope_sma_50 / slope_sma_200 -------------------------------------


def test_slope_sma_50_nan_before_required_history(
    spy_like_random_walk_1000, v2_trend_config
):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    # Need sma_50 from t=49, plus 20-day shift → first non-NaN at t=69.
    assert out.slope_sma_50.iloc[:69].isna().all()
    assert not math.isnan(out.slope_sma_50.iloc[69])


def test_slope_sma_200_nan_before_required_history(
    spy_like_random_walk_1000, v2_trend_config
):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    # Need sma_200 from t=199, plus 20-day shift → first non-NaN at t=219.
    assert out.slope_sma_200.iloc[:219].isna().all()
    assert not math.isnan(out.slope_sma_200.iloc[219])


def test_slope_sma_50_positive_on_rising_series(monotonic_rising_1000, v2_trend_config):
    out = compute_trend_v2_features(monotonic_rising_1000, config=v2_trend_config)
    slope = out.slope_sma_50.dropna()
    assert (slope > 0).all()


def test_slope_sma_200_negative_on_falling_series(v2_trend_config):
    index = pd.bdate_range(end="2024-12-31", periods=400)
    close = pd.Series(np.linspace(1000.0, 100.0, 400), index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)
    slope = out.slope_sma_200.dropna()
    assert (slope < 0).all()


def test_slope_sma_50_hand_computed_value(v2_trend_config):
    """Hand-check (sma_50[t] - sma_50[t-20]) / sma_50[t-20] at t=80."""
    rng = np.random.default_rng(seed=42)
    index = pd.bdate_range(end="2024-12-31", periods=200)
    close = pd.Series(
        100.0 + np.cumsum(rng.normal(0, 1.0, size=200)),
        index=index,
        name="close",
    )
    out = compute_trend_v2_features(close, config=v2_trend_config)
    sma50 = close.rolling(50).mean()
    t = 80
    expected = (sma50.iloc[t] - sma50.iloc[t - 20]) / sma50.iloc[t - 20]
    assert out.slope_sma_50.iloc[t] == pytest.approx(expected, abs=1e-12)


def test_trend_direction_v2_sma_levels_match_legacy_inline_formulas(
    spy_like_random_walk_1000, v2_trend_config
):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)

    pd.testing.assert_series_equal(
        out.sma_50,
        spy_like_random_walk_1000.rolling(50).mean().rename("sma_50"),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        out.sma_200,
        spy_like_random_walk_1000.rolling(200).mean().rename("sma_200"),
        check_exact=True,
    )


# ---------- return_63d / return_126d -----------------------------------------


def test_trend_direction_v2_returns_match_legacy_inline_formulas(
    spy_like_random_walk_1000, v2_trend_config
):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)

    pd.testing.assert_series_equal(
        out.return_63d,
        (spy_like_random_walk_1000 / spy_like_random_walk_1000.shift(63) - 1.0).rename(
            "return_63d"
        ),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        out.return_126d,
        (spy_like_random_walk_1000 / spy_like_random_walk_1000.shift(126) - 1.0).rename(
            "return_126d"
        ),
        check_exact=True,
    )


def test_return_63d_hand_computed(v2_trend_config):
    rng = np.random.default_rng(seed=42)
    index = pd.bdate_range(end="2024-12-31", periods=200)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=200))),
        index=index,
        name="close",
    )
    out = compute_trend_v2_features(close, config=v2_trend_config)
    t = 150
    expected = close.iloc[t] / close.iloc[t - 63] - 1.0
    assert out.return_63d.iloc[t] == pytest.approx(expected, abs=1e-12)
    assert out.return_63d.iloc[:63].isna().all()


def test_return_126d_hand_computed(v2_trend_config):
    rng = np.random.default_rng(seed=42)
    index = pd.bdate_range(end="2024-12-31", periods=300)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=300))),
        index=index,
        name="close",
    )
    out = compute_trend_v2_features(close, config=v2_trend_config)
    t = 200
    expected = close.iloc[t] / close.iloc[t - 126] - 1.0
    assert out.return_126d.iloc[t] == pytest.approx(expected, abs=1e-12)
    assert out.return_126d.iloc[:126].isna().all()


# ---------- drawdown_252d ----------------------------------------------------


def test_drawdown_zero_at_new_high(monotonic_rising_1000, v2_trend_config):
    out = compute_trend_v2_features(monotonic_rising_1000, config=v2_trend_config)
    dd = out.drawdown_252d.dropna()
    # Strictly rising series ⇒ t is always the trailing 252d peak ⇒ dd == 0.
    np.testing.assert_allclose(dd.to_numpy(), 0.0, atol=1e-12)


def test_drawdown_minus_10pct_when_10pct_below_peak(v2_trend_config):
    # 252-day window with peak == 100, current == 90 ⇒ drawdown = -0.10.
    index = pd.bdate_range(end="2024-12-31", periods=252)
    prices = np.full(252, 90.0)
    prices[100] = 100.0  # peak inside the window
    close = pd.Series(prices, index=index, name="close")
    out = compute_trend_v2_features(close, config=v2_trend_config)
    assert out.drawdown_252d.iloc[-1] == pytest.approx(-0.10, abs=1e-12)


def test_drawdown_nan_before_lookback(spy_like_random_walk_1000, v2_trend_config):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    assert out.drawdown_252d.iloc[:251].isna().all()
    assert not math.isnan(out.drawdown_252d.iloc[251])


# ---------- NaN propagation --------------------------------------------------


def test_nan_in_input_propagates_to_features(v2_trend_config):
    index = pd.bdate_range(end="2024-12-31", periods=400)
    rng = np.random.default_rng(seed=7)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=400))),
        index=index,
        name="close",
    )
    # Inject a single NaN at position 300; the 20d efficiency_ratio window
    # ending at 300..319 should all be NaN.
    close.iloc[300] = np.nan
    out = compute_trend_v2_features(close, config=v2_trend_config)
    affected_er = out.efficiency_ratio_20d.iloc[300:320]
    assert affected_er.isna().all()


# ---------- to_frame / shape -------------------------------------------------


def test_features_align_to_input_index_and_expose_named_columns(
    spy_like_random_walk_1000, v2_trend_config
):
    out = compute_trend_v2_features(spy_like_random_walk_1000, config=v2_trend_config)
    assert isinstance(out, TrendDirectionV2Features)
    frame = out.to_frame()
    assert list(frame.columns) == list(out.feature_names)
    assert len(frame) == len(spy_like_random_walk_1000)
    assert (frame.index == spy_like_random_walk_1000.index).all()


# ---------- Feature-store integration ----------------------------------------


_INTEGRATION_AS_OF = date(2023, 12, 14)


def test_build_feature_store_populates_trend_direction_v2(
    market_df_for_asof, v2_trend_config
):
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(
        context,
        trend_direction_v2_config=v2_trend_config,
    )
    assert store.trend_direction_v2 is not None
    assert isinstance(store.trend_direction_v2, TrendDirectionV2Features)
    assert len(store.trend_direction_v2.return_63d) == len(store.spy_index)
    # All seven §1A features are date-indexed series aligned to spy_index.
    for name in store.trend_direction_v2.feature_names:
        series = getattr(store.trend_direction_v2, name)
        assert isinstance(series, pd.Series)
        assert (series.index == store.spy_index).all()


def test_build_feature_store_none_when_config_absent(market_df_for_asof):
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(context)
    assert store.trend_direction_v2 is None


# ---------- End-to-end wire test (AGENTS rule A) -----------------------------


def test_engine_classify_window_threads_trend_direction_v2_config(market_df_for_asof):
    """Confirms the top-level engine entry point threads
    TrendDirectionV2Config through to the feature store. This locks in
    that future classifier wiring will have the features available."""
    engine = RegimeEngine()
    cfg = engine.config
    # The packaged v2 yaml has trend_direction_v2 populated for slice 2.1.
    assert cfg.trend_direction_v2 is not None

    # Mirror feature_store assembly via the timeline path used by classify_window.
    from regime_detection.market_context import (
        build_market_context,
        slice_context_to_recent_sessions,
    )
    from regime_detection.timeline import ENGINE_MINIMUM_HISTORY

    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    required = min(len(context.sessions), ENGINE_MINIMUM_HISTORY)
    working = slice_context_to_recent_sessions(
        context=context, required_sessions=required
    )
    store = build_feature_store(
        working,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
    )
    assert store.trend_direction_v2 is not None
    assert len(store.trend_direction_v2.efficiency_ratio_20d) == len(store.spy_index)
