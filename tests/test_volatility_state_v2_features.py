"""TDD tests for v2 §1C Layer 1 V2 Volatility features (Slice 2.2).

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic SPY-like OHLC series
and the real production Pydantic config — NO toy a/b/c names. Math is
verified against hand-computed values.

Spec references:
    docs/regime_engine_v2_spec.md §1C (lines 138–192).
    Slice scope: features only (no rising_vol/vol_crush labels yet); IV/RV
    and vol_crush features deferred (require options data). See §8 line 1181.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import (
    VolatilityV2Config,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.timeline import build_regime_timeline
from regime_detection.volatility_state import wilders_atr
from regime_detection.volatility_state_v2 import (
    VolatilityV2Features,
    compute_volatility_v2_features,
)


# ---------- Shared fixtures ---------------------------------------------------


@pytest.fixture
def v2_volatility_config() -> VolatilityV2Config:
    """Real production defaults from configs/core3-v2.0.0.yaml."""
    return VolatilityV2Config(
        atr_short_period=14,
        atr_long_period=50,
        gap_frequency_lookback_days=20,
        gap_threshold_pct=0.005,
        intraday_range_lookback_days=252,
    )


def _index_n(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2024-12-31", periods=n)


@pytest.fixture
def spy_like_ohlc_1000() -> dict[str, pd.Series]:
    """1000-session synthetic SPY-like OHLC, GBM-style. Seed pinned."""
    rng = np.random.default_rng(seed=20260512)
    n = 1000
    log_rets = rng.normal(loc=0.0003, scale=0.01, size=n)
    close_arr = 400.0 * np.exp(np.cumsum(log_rets))
    # high/low/open derived from a small intraday-vol band so the series
    # looks like real SPY OHLC rather than synthetic zero-range bars.
    intraday_vol = rng.uniform(0.003, 0.015, size=n)
    rng2 = np.random.default_rng(seed=99)
    open_arr = close_arr * (1.0 + rng2.normal(0, 0.001, size=n))
    high_arr = np.maximum(open_arr, close_arr) * (1.0 + intraday_vol)
    low_arr = np.minimum(open_arr, close_arr) * (1.0 - intraday_vol)
    index = _index_n(n)
    return {
        "open": pd.Series(open_arr, index=index, name="open"),
        "high": pd.Series(high_arr, index=index, name="high"),
        "low": pd.Series(low_arr, index=index, name="low"),
        "close": pd.Series(close_arr, index=index, name="close"),
    }


# =============================================================================
# wilders_atr — shared v1/v2 helper
# =============================================================================


def test_wilders_atr_hand_computed_seed_and_step():
    """Hand-compute Wilder's ATR_3 on a 5-day OHLC fixture."""
    index = _index_n(5)
    high = pd.Series([10.0, 11.0, 12.5, 13.0, 12.8], index=index)
    low = pd.Series([9.0, 10.0, 11.0, 11.5, 11.0], index=index)
    close = pd.Series([9.5, 10.8, 12.0, 12.5, 11.8], index=index)
    atr = wilders_atr(high=high, low=low, close=close, period=3)
    # TR series:
    # t=0: high-low = 1.0
    # t=1: max(11-10, |11-9.5|, |10-9.5|) = max(1, 1.5, 0.5) = 1.5
    # t=2: max(12.5-11, |12.5-10.8|, |11-10.8|) = max(1.5, 1.7, 0.2) = 1.7
    # t=3: max(13-11.5, |13-12|, |11.5-12|) = max(1.5, 1.0, 0.5) = 1.5
    # t=4: max(12.8-11, |12.8-12.5|, |11-12.5|) = max(1.8, 0.3, 1.5) = 1.8
    # Seed at t=2: mean(TR[0..2]) = (1.0 + 1.5 + 1.7) / 3 = 4.2 / 3 = 1.4
    assert atr.iloc[0:2].isna().all()
    assert atr.iloc[2] == pytest.approx(1.4, abs=1e-12)
    # t=3: (1.4 * 2 + 1.5) / 3 = (2.8 + 1.5) / 3 = 4.3/3
    assert atr.iloc[3] == pytest.approx(4.3 / 3.0, abs=1e-12)
    # t=4: (atr[3] * 2 + 1.8) / 3
    expected = (atr.iloc[3] * 2 + 1.8) / 3.0
    assert atr.iloc[4] == pytest.approx(expected, abs=1e-12)


def test_wilders_atr_constant_ohlc_is_zero_after_seed():
    """Constant OHLC (zero true range) → ATR stays at 0 after seed."""
    index = _index_n(50)
    constant = pd.Series(np.full(50, 100.0), index=index)
    atr = wilders_atr(high=constant, low=constant, close=constant, period=14)
    assert atr.iloc[:13].isna().all()
    np.testing.assert_allclose(atr.iloc[13:].to_numpy(), 0.0, atol=1e-12)


def test_wilders_atr_rejects_bad_period():
    index = _index_n(5)
    s = pd.Series(np.ones(5), index=index)
    with pytest.raises(ValueError):
        wilders_atr(high=s, low=s, close=s, period=0)


# =============================================================================
# atr_ratio
# =============================================================================


def test_atr_ratio_nan_before_long_lookback(
    spy_like_ohlc_1000, v2_volatility_config
):
    out = compute_volatility_v2_features(
        open_=spy_like_ohlc_1000["open"],
        high=spy_like_ohlc_1000["high"],
        low=spy_like_ohlc_1000["low"],
        close=spy_like_ohlc_1000["close"],
        config=v2_volatility_config,
    )
    # Long ATR (50) seeds at t=49; ratio NaN before that.
    assert out.atr_ratio.iloc[:49].isna().all()
    assert not math.isnan(out.atr_ratio.iloc[49])


def test_atr_ratio_constant_ohlc_is_nan(v2_volatility_config):
    """Constant OHLC → both ATRs are 0 → ratio is NaN (0/0)."""
    index = _index_n(100)
    constant = pd.Series(np.full(100, 250.0), index=index)
    out = compute_volatility_v2_features(
        open_=constant,
        high=constant,
        low=constant,
        close=constant,
        config=v2_volatility_config,
    )
    # ATR_50 == 0 → ratio masked to NaN.
    assert out.atr_ratio.iloc[49:].isna().all()


def test_atr_ratio_rising_vol_above_one(v2_volatility_config):
    """Late-window vol expansion → ATR_14 > ATR_50 → ratio > 1.0."""
    n = 300
    index = _index_n(n)
    rng = np.random.default_rng(seed=12345)
    # Low-vol prefix, then high-vol suffix in the last 30 days.
    intraday = np.concatenate(
        [
            rng.uniform(0.001, 0.002, size=n - 30),
            rng.uniform(0.03, 0.05, size=30),
        ]
    )
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, size=n)))
    open_ = close * (1.0 + rng.normal(0, 0.0005, size=n))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    out = compute_volatility_v2_features(
        open_=pd.Series(open_, index=index),
        high=pd.Series(high, index=index),
        low=pd.Series(low, index=index),
        close=pd.Series(close, index=index),
        config=v2_volatility_config,
    )
    assert out.atr_ratio.iloc[-1] > 1.0


def test_atr_ratio_falling_vol_below_one(v2_volatility_config):
    """High-vol prefix → low-vol suffix → ATR_14 < ATR_50 → ratio < 1.0."""
    n = 300
    index = _index_n(n)
    rng = np.random.default_rng(seed=54321)
    intraday = np.concatenate(
        [
            rng.uniform(0.03, 0.05, size=n - 30),
            rng.uniform(0.001, 0.002, size=30),
        ]
    )
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, size=n)))
    open_ = close * (1.0 + rng.normal(0, 0.0005, size=n))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    out = compute_volatility_v2_features(
        open_=pd.Series(open_, index=index),
        high=pd.Series(high, index=index),
        low=pd.Series(low, index=index),
        close=pd.Series(close, index=index),
        config=v2_volatility_config,
    )
    assert out.atr_ratio.iloc[-1] < 1.0


# =============================================================================
# gap_frequency_20d
# =============================================================================


def test_gap_frequency_zero_when_no_gaps(v2_volatility_config):
    """open[t] == close[t-1] everywhere → gap == 0 → frequency == 0."""
    n = 50
    index = _index_n(n)
    close = pd.Series(100.0 + np.arange(n) * 0.01, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = open_.combine(close, max) + 0.1
    low = open_.combine(close, min) - 0.1
    out = compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=v2_volatility_config,
    )
    # After 20 sessions of zero gaps, frequency == 0.
    valid = out.gap_frequency_20d.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.to_numpy(), 0.0, atol=1e-12)


def test_gap_frequency_one_when_every_day_gaps(v2_volatility_config):
    """Every day has gap > 0.005 → frequency == 1.0 once warmed."""
    n = 50
    index = _index_n(n)
    close = pd.Series(np.full(n, 100.0), index=index)
    # open[t] = close[t-1] * 1.02 → gap = 0.02 > threshold every day.
    open_ = pd.Series(np.full(n, 102.0), index=index)
    high = pd.Series(np.full(n, 103.0), index=index)
    low = pd.Series(np.full(n, 99.0), index=index)
    out = compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=v2_volatility_config,
    )
    # First non-NaN frequency at t=19 (need 20 obs incl t and one prev close).
    valid = out.gap_frequency_20d.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 1.0, atol=1e-12)


def test_gap_frequency_hand_computed_4_of_20(v2_volatility_config):
    """25-day fixture: exactly 4 gaps > 0.005 in window [t-19..t], rest at 0.0.
    Expected frequency at t=24: 4 / 20 = 0.2.
    """
    n = 25
    index = _index_n(n)
    # close = 100 every day → prev_close = 100.
    close = pd.Series(np.full(n, 100.0), index=index)
    open_arr = np.full(n, 100.0)  # gap = 0 by default
    # Place 4 gaps within the last 20 sessions (indices 6..24 inclusive).
    # t=5 is index 5 (NOT in the last-20 window which ends at t=24
    # and starts at t-19 = 5). So index 5 IS included.
    # Use indices 10, 15, 20, 23 (all within [5,24]).
    for i in [10, 15, 20, 23]:
        open_arr[i] = 100.6  # gap = 0.006 > 0.005
    open_ = pd.Series(open_arr, index=index)
    high = pd.Series(np.full(n, 101.0), index=index)
    low = pd.Series(np.full(n, 99.0), index=index)
    out = compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=v2_volatility_config,
    )
    assert out.gap_frequency_20d.iloc[-1] == pytest.approx(4.0 / 20.0, abs=1e-12)


def test_gap_frequency_boundary_exactly_at_threshold_not_counted(
    v2_volatility_config,
):
    """Gap == 0.005 exactly → strictly-greater rule excludes it."""
    n = 30
    index = _index_n(n)
    close = pd.Series(np.full(n, 100.0), index=index)
    # Every gap exactly 0.5%: open = 100 * 1.005 = 100.5.
    open_ = pd.Series(np.full(n, 100.5), index=index)
    high = pd.Series(np.full(n, 101.0), index=index)
    low = pd.Series(np.full(n, 99.0), index=index)
    out = compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=v2_volatility_config,
    )
    valid = out.gap_frequency_20d.dropna()
    np.testing.assert_allclose(valid.to_numpy(), 0.0, atol=1e-12)


def test_gap_frequency_nan_before_lookback(
    spy_like_ohlc_1000, v2_volatility_config
):
    out = compute_volatility_v2_features(
        open_=spy_like_ohlc_1000["open"],
        high=spy_like_ohlc_1000["high"],
        low=spy_like_ohlc_1000["low"],
        close=spy_like_ohlc_1000["close"],
        config=v2_volatility_config,
    )
    # Need 20 gap observations, and gap[0] is NaN (no prev close), so the
    # first non-NaN frequency is at t = 20 (window covers t=1..t=20).
    assert out.gap_frequency_20d.iloc[:20].isna().all()
    assert not math.isnan(out.gap_frequency_20d.iloc[20])


# =============================================================================
# intraday_range_percentile_252d
# =============================================================================


def test_intraday_range_percentile_constant_range(v2_volatility_config):
    """Constant (high-low)/close → all 252d rolling ranks are identical
    (every observation ties; default `method='average'` puts the rank at
    the midpoint = (N + 1) / (2N) ≈ 0.5).
    """
    n = 300
    index = _index_n(n)
    close = pd.Series(np.full(n, 100.0), index=index)
    open_ = pd.Series(np.full(n, 100.0), index=index)
    high = pd.Series(np.full(n, 102.0), index=index)
    low = pd.Series(np.full(n, 98.0), index=index)
    out = compute_volatility_v2_features(
        open_=open_, high=high, low=low, close=close, config=v2_volatility_config
    )
    valid = out.intraday_range_percentile_252d.dropna()
    assert len(valid) > 0
    # All values are tied; pandas average-tie rank pct = (N+1)/(2N) ≈ 0.502.
    expected = (252.0 + 1.0) / (2.0 * 252.0)
    np.testing.assert_allclose(valid.to_numpy(), expected, atol=1e-12)


def test_intraday_range_percentile_rises_with_range(v2_volatility_config):
    """A monotonically rising intraday range → percentile → 1.0 at the end."""
    n = 300
    index = _index_n(n)
    close = pd.Series(np.full(n, 100.0), index=index)
    open_ = pd.Series(np.full(n, 100.0), index=index)
    # range = i * 0.01: strictly increasing.
    range_arr = (np.arange(n) + 1) * 0.01
    high = pd.Series(100.0 + range_arr / 2.0, index=index)
    low = pd.Series(100.0 - range_arr / 2.0, index=index)
    out = compute_volatility_v2_features(
        open_=open_, high=high, low=low, close=close, config=v2_volatility_config
    )
    # Last value is the max within its 252d window → percentile == 1.0.
    assert out.intraday_range_percentile_252d.iloc[-1] == pytest.approx(
        1.0, abs=1e-12
    )


def test_intraday_range_percentile_nan_before_lookback(
    spy_like_ohlc_1000, v2_volatility_config
):
    out = compute_volatility_v2_features(
        open_=spy_like_ohlc_1000["open"],
        high=spy_like_ohlc_1000["high"],
        low=spy_like_ohlc_1000["low"],
        close=spy_like_ohlc_1000["close"],
        config=v2_volatility_config,
    )
    assert out.intraday_range_percentile_252d.iloc[:251].isna().all()
    assert not math.isnan(out.intraday_range_percentile_252d.iloc[251])


# =============================================================================
# Shape / to_frame
# =============================================================================


def test_features_align_to_input_index(spy_like_ohlc_1000, v2_volatility_config):
    out = compute_volatility_v2_features(
        open_=spy_like_ohlc_1000["open"],
        high=spy_like_ohlc_1000["high"],
        low=spy_like_ohlc_1000["low"],
        close=spy_like_ohlc_1000["close"],
        config=v2_volatility_config,
    )
    assert isinstance(out, VolatilityV2Features)
    frame = out.to_frame()
    assert list(frame.columns) == list(out.feature_names)
    assert len(frame) == len(spy_like_ohlc_1000["close"])
    assert (frame.index == spy_like_ohlc_1000["close"].index).all()


# =============================================================================
# Feature-store + timeline integration (AGENTS rule A)
# =============================================================================


_INTEGRATION_AS_OF = date(2023, 12, 14)


def test_build_feature_store_populates_volatility_state_v2(
    market_df_for_asof, v2_volatility_config
):
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(
        context, volatility_state_v2_config=v2_volatility_config
    )
    assert store.volatility_state_v2 is not None
    assert isinstance(store.volatility_state_v2, VolatilityV2Features)
    for name in store.volatility_state_v2.feature_names:
        series = getattr(store.volatility_state_v2, name)
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
    assert store.volatility_state_v2 is None


def test_timeline_threads_volatility_state_v2_config(market_df_for_asof):
    """End-to-end wire test: build_regime_timeline must accept v2 config and
    surface volatility_state_v2 features via the same feature_store path used
    by the engine. This locks in that future classifier wiring will have
    the features available."""
    engine = RegimeEngine()
    cfg = engine.config
    assert cfg.volatility_state_v2 is not None

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
    )
    assert store.volatility_state_v2 is not None
    assert len(store.volatility_state_v2.atr_ratio) == len(store.spy_index)

    # build_regime_timeline must propagate the v2 config without raising
    # (no v1 contract drift — RegimeTimeline doesn't expose the v2 features
    # yet; slice 2.2 only ships compute + seam).
    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg
    )
    assert len(timeline.outputs) == 5


def test_v1_config_path_leaves_volatility_state_v2_none(market_df_for_asof):
    """When volatility_state_v2 is None the feature store's volatility_state_v2
    is None. Per-label hysteresis is required in the axis builder, so the
    timeline cannot build without the config — this test only verifies the
    feature_store seam is correctly absent."""
    cfg = load_default_regime_config()
    cfg_v1 = cfg.model_copy(update={"volatility_state_v2": None})
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg_v1,
    )
    store = build_feature_store(context)
    assert store.volatility_state_v2 is None
