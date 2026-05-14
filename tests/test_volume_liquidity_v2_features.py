"""TDD tests for v2 §1E Layer 1 V2 Volume / Liquidity features (Slice 2.4).

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic SPY-scale volume
(tens of millions of shares) and the real production Pydantic config —
NO toy a/b/c names. Math is verified against hand-computed values.

Spec references:
    docs/regime_engine_v2_spec.md §1E (lines 251–294).
    This feature module owns ``volume_zscore_20d`` only. The §1E labels,
    rule engine, risk-rank table, and hysteresis are covered by the
    volume-liquidity classifier/rule tests.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import (
    VolumeLiquidityV2Config,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.timeline import build_regime_timeline
from regime_detection.volume_liquidity_v2 import (
    VolumeLiquidityV2Features,
    compute_volume_liquidity_v2_features,
)


# ---------- Shared fixtures ---------------------------------------------------


@pytest.fixture
def v2_volume_config() -> VolumeLiquidityV2Config:
    """Real production defaults from configs/core3-v2.0.0.yaml."""
    return VolumeLiquidityV2Config(
        volume_zscore_lookback_days=20,
        volume_zscore_ddof=1,
    )


def _index_n(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2024-12-31", periods=n)


# Realistic SPY-volume scale: ~80M shares/day mean, ~20M std.
_SPY_VOLUME_MEAN = 80_000_000.0
_SPY_VOLUME_STD = 20_000_000.0


@pytest.fixture
def spy_like_volume_1000() -> pd.Series:
    """1000-session synthetic SPY-volume series. Seed pinned."""
    rng = np.random.default_rng(seed=20260512)
    n = 1000
    # Truncated normal-ish via abs() to keep volumes positive at SPY scale.
    volume = np.abs(
        rng.normal(loc=_SPY_VOLUME_MEAN, scale=_SPY_VOLUME_STD, size=n)
    )
    return pd.Series(volume, index=_index_n(n), name="volume")


# =============================================================================
# volume_zscore_20d
# =============================================================================


def test_volume_zscore_constant_volume_is_nan(v2_volume_config):
    """Constant volume → rolling std == 0 → z-score is NaN (0/0)."""
    n = 50
    index = _index_n(n)
    constant_volume = pd.Series(
        np.full(n, _SPY_VOLUME_MEAN), index=index, name="volume"
    )
    out = compute_volume_liquidity_v2_features(
        volume=constant_volume, config=v2_volume_config
    )
    # After warm-up at t=19, std is zero → NaN.
    assert out.volume_zscore_20d.iloc[19:].isna().all()


def test_volume_zscore_nan_before_lookback(spy_like_volume_1000, v2_volume_config):
    """`min_periods=20` ⇒ first valid index is t=19 (20 obs in [0..19])."""
    out = compute_volume_liquidity_v2_features(
        volume=spy_like_volume_1000, config=v2_volume_config
    )
    assert out.volume_zscore_20d.iloc[:19].isna().all()
    assert not math.isnan(out.volume_zscore_20d.iloc[19])


def test_volume_zscore_hand_computed_25day_fixture(v2_volume_config):
    """Hand-compute the z-score on a 25-day SPY-scale fixture at t=24.

    Volume sequence: first 19 days vary around 80M, day 24 is a known
    value. Z-score = (v[24] - mean(v[5..24])) / std(v[5..24], ddof=1).
    """
    n = 25
    index = _index_n(n)
    # Days 0..23: stable mean=80M, alternating ±10M. Day 24: spike to 130M.
    base = np.full(n, _SPY_VOLUME_MEAN)
    base[1::2] += 10_000_000.0
    base[::2] -= 10_000_000.0
    base[24] = 130_000_000.0
    volume = pd.Series(base, index=index, name="volume")
    out = compute_volume_liquidity_v2_features(
        volume=volume, config=v2_volume_config
    )
    window = volume.iloc[5:25]  # 20 obs ending at t=24
    expected = (volume.iloc[24] - window.mean()) / window.std(ddof=1)
    assert out.volume_zscore_20d.iloc[24] == pytest.approx(expected, rel=1e-12)


def test_volume_zscore_spike_day_approx_plus_5_sigma(v2_volume_config):
    """20 days of (near-) constant volume + 1 anomalous +5σ day → z ≈ +5.

    Use a tiny jitter so std > 0 (and the resulting std equals the
    jitter scale). Then place a +5σ spike on the final day and verify.
    """
    n = 21
    index = _index_n(n)
    rng = np.random.default_rng(seed=42)
    jitter_scale = 1_000_000.0
    base = _SPY_VOLUME_MEAN + rng.normal(0, jitter_scale, size=n)
    base[20] = _SPY_VOLUME_MEAN + 5.0 * jitter_scale  # spike — but recompute
    volume = pd.Series(base, index=index, name="volume")
    out = compute_volume_liquidity_v2_features(
        volume=volume, config=v2_volume_config
    )
    # The window at t=20 covers t=1..t=20 inclusive.
    window = volume.iloc[1:21]
    expected = (volume.iloc[20] - window.mean()) / window.std(ddof=1)
    assert out.volume_zscore_20d.iloc[20] == pytest.approx(expected, rel=1e-12)
    # Sanity: a ~5σ spike against the jittered baseline gives a
    # z-score well above 3 (we don't pin exactly 5 because the spike
    # itself shifts the window's mean and std).
    assert out.volume_zscore_20d.iloc[20] > 3.0


def test_volume_zscore_boundary_below_2_sigma_panic_threshold(v2_volume_config):
    """Future labels slice rule (§1E line 272): `panic_volume` requires
    `volume_zscore_20d > 2.0`. Verify a day engineered to land just
    below 2.0 produces a value < 2.0 (the rule itself is NOT applied
    here — feature only)."""
    n = 21
    index = _index_n(n)
    jitter_scale = 1_000_000.0
    base = np.full(n, _SPY_VOLUME_MEAN, dtype=float)
    # Pre-warm the window with tiny alternating jitter so std > 0.
    base[1::2] += jitter_scale
    base[::2] -= jitter_scale
    # Day 20: 1.5σ-ish spike above the mean.
    base[20] = _SPY_VOLUME_MEAN + 1.5 * jitter_scale
    volume = pd.Series(base, index=index, name="volume")
    out = compute_volume_liquidity_v2_features(
        volume=volume, config=v2_volume_config
    )
    z_at_t = out.volume_zscore_20d.iloc[20]
    assert z_at_t < 2.0
    assert z_at_t > 0.0


# =============================================================================
# Shape / to_frame
# =============================================================================


def test_features_align_to_input_index(spy_like_volume_1000, v2_volume_config):
    out = compute_volume_liquidity_v2_features(
        volume=spy_like_volume_1000, config=v2_volume_config
    )
    assert isinstance(out, VolumeLiquidityV2Features)
    frame = out.to_frame()
    assert list(frame.columns) == list(out.feature_names)
    assert len(frame) == len(spy_like_volume_1000)
    assert (frame.index == spy_like_volume_1000.index).all()


# =============================================================================
# Feature-store + timeline integration (AGENTS rule A)
# =============================================================================


_INTEGRATION_AS_OF = date(2023, 12, 14)


def test_build_feature_store_populates_volume_liquidity_v2(
    market_df_for_asof, v2_volume_config
):
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(
        context, volume_liquidity_v2_config=v2_volume_config
    )
    assert store.volume_liquidity_v2 is not None
    assert isinstance(store.volume_liquidity_v2, VolumeLiquidityV2Features)
    series = store.volume_liquidity_v2.volume_zscore_20d
    assert isinstance(series, pd.Series)
    assert (series.index == store.spy_index).all()
    # At least one non-NaN value should land for a real ~1000-session SPY history.
    assert series.notna().any()


def test_build_feature_store_none_when_config_absent(market_df_for_asof):
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(context)
    assert store.volume_liquidity_v2 is None


def test_timeline_threads_volume_liquidity_v2_config(market_df_for_asof):
    """End-to-end wire test (AGENTS rule A): build_regime_timeline must
    accept the v2 config and surface volume_liquidity_v2 features via
    the same feature_store path used by the engine. This locks in
    that future classifier wiring will have the feature available."""
    engine = RegimeEngine()
    cfg = engine.config
    assert cfg.volume_liquidity_v2 is not None

    from regime_detection.market_context import (
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
        volatility_state_v2_config=cfg.volatility_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
    )
    assert store.volume_liquidity_v2 is not None
    assert len(store.volume_liquidity_v2.volume_zscore_20d) == len(
        store.spy_index
    )

    # build_regime_timeline must propagate the v2 config without raising.
    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg
    )
    assert len(timeline.outputs) == 5


def test_v1_config_path_leaves_volume_liquidity_v2_none(market_df_for_asof):
    """V1 contract preservation: loading a v1-only config (no v2 sub-blocks)
    yields a feature store where volume_liquidity_v2 is None and the
    timeline builds without raising."""
    cfg = load_default_regime_config()
    cfg_v1 = cfg.model_copy(update={"volume_liquidity_v2": None})
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg_v1,
    )
    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg_v1
    )
    assert len(timeline.outputs) == 5
    store = build_feature_store(context)
    assert store.volume_liquidity_v2 is None
