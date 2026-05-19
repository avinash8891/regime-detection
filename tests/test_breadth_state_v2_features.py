"""Tests for v2 §1D Layer 1 V2 Breadth features.

Per ~/.claude/CLAUDE.md and AGENTS rule G/L: real production sector symbols
imported from `SECTOR_ETFS` — NO toy names.

Spec references:
    docs/regime_engine_v2_spec.md §1D (lines 196–247).
    Sector breadth plus optional PIT-derived features / labels.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.breadth_state_v2 import (
    BreadthV2Features,
    compute_breadth_v2_features,
)
from regime_detection.config import (
    BreadthV2Config,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.market_context import build_market_context
from regime_detection.timeline import build_regime_timeline


# ---------- Shared fixtures ---------------------------------------------------


@pytest.fixture
def v2_breadth_config() -> BreadthV2Config:
    """Real production default from configs/core3-v2.0.0.yaml."""
    return BreadthV2Config(sector_breadth_lookback_days=21)


def _index_n(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2024-12-31", periods=n)


def _sector_closes(
    *, n: int, return_signs: dict[str, int], lookback: int = 21
) -> dict[str, pd.Series]:
    """Build a dict of sector closes such that ``close[t] / close[t-lookback] - 1``
    has the sign specified per sector.

    ``return_signs[symbol]`` ∈ {+1, 0, -1}:
      +1 → strictly positive 21d return (every session past warmup).
       0 → exactly 0.
      -1 → strictly negative.
    """
    index = _index_n(n)
    out: dict[str, pd.Series] = {}
    for symbol in SECTOR_ETFS:
        sign = return_signs[symbol]
        if sign > 0:
            # Monotone-rising series → return > 0 everywhere past warmup.
            arr = 100.0 * (1.0 + np.arange(n) * 0.001)
        elif sign < 0:
            arr = 100.0 * (1.0 - np.arange(n) * 0.001)
        else:
            # Constant series → return == 0 everywhere past warmup.
            arr = np.full(n, 100.0)
        out[symbol] = pd.Series(arr, index=index, name=symbol)
    return out


# =============================================================================
# sector_breadth boundary tests
# =============================================================================


def test_sector_breadth_all_positive_is_one(v2_breadth_config):
    closes = _sector_closes(
        n=60, return_signs={s: +1 for s in SECTOR_ETFS}
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    # After warmup (t >= 21), every sector has return > 0 → breadth = 11/11 = 1.0.
    assert out.sector_breadth.iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert out.sector_breadth.iloc[30] == pytest.approx(1.0, abs=1e-12)


def test_sector_breadth_all_negative_is_zero(v2_breadth_config):
    closes = _sector_closes(
        n=60, return_signs={s: -1 for s in SECTOR_ETFS}
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    assert out.sector_breadth.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_sector_breadth_five_positive_six_nonpositive(v2_breadth_config):
    """5 sectors with positive return, 3 negative, 3 exactly zero → 5/11."""
    positives = {"XLB", "XLC", "XLE", "XLF", "XLI"}
    negatives = {"XLK", "XLP", "XLRE"}
    zeros = {"XLU", "XLV", "XLY"}
    signs = {}
    for s in SECTOR_ETFS:
        if s in positives:
            signs[s] = +1
        elif s in negatives:
            signs[s] = -1
        else:
            assert s in zeros
            signs[s] = 0
    closes = _sector_closes(n=60, return_signs=signs)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    assert out.sector_breadth.iloc[-1] == pytest.approx(5.0 / 11.0, abs=1e-12)


def test_sector_breadth_boundary_exactly_zero_not_counted(v2_breadth_config):
    """All 11 sectors with return_21d == 0.0 → breadth = 0.0
    (strictly `> 0` rule from §1D line 229)."""
    closes = _sector_closes(
        n=60, return_signs={s: 0 for s in SECTOR_ETFS}
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    valid = out.sector_breadth.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.to_numpy(), 0.0, atol=1e-12)


def test_sector_breadth_hand_computed_30d_synthetic(v2_breadth_config):
    """30-day fixture with hand-computed sector returns.

    Constructed so on the final session:
      - XLB, XLC, XLE: close[t] = 110, close[t-21] = 100  → +10% (>0)
      - XLF, XLI:      close[t] = 105, close[t-21] = 100  → +5%  (>0)
      - XLK, XLP, XLRE, XLU: close[t] = 95, close[t-21] = 100 → -5% (<0)
      - XLV, XLY:      close[t] = 100, close[t-21] = 100  →  0%  (NOT counted)

    Expected sector_breadth = 5/11.
    """
    n = 30
    index = _index_n(n)
    lookback = v2_breadth_config.sector_breadth_lookback_days
    # warmup of 21 sessions at 100, then ramp to the target close on day 21+
    final_closes_by_symbol = {
        "XLB": 110.0,
        "XLC": 110.0,
        "XLE": 110.0,
        "XLF": 105.0,
        "XLI": 105.0,
        "XLK": 95.0,
        "XLP": 95.0,
        "XLRE": 95.0,
        "XLU": 95.0,
        "XLV": 100.0,
        "XLY": 100.0,
    }
    closes: dict[str, pd.Series] = {}
    for symbol in SECTOR_ETFS:
        target = final_closes_by_symbol[symbol]
        arr = np.full(n, 100.0)
        # Linearly interpolate from 100 (at t=lookback-1, i.e. the prior-close
        # reference for the final session) to target at t=n-1.
        ramp_start_t = lookback - 1
        for i in range(ramp_start_t, n):
            frac = (i - ramp_start_t) / (n - 1 - ramp_start_t)
            arr[i] = 100.0 + (target - 100.0) * frac
        closes[symbol] = pd.Series(arr, index=index, name=symbol)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    assert out.sector_breadth.iloc[-1] == pytest.approx(5.0 / 11.0, abs=1e-12)


def test_sector_breadth_nan_before_lookback(v2_breadth_config):
    """For t < lookback the 21d return is NaN → sector_breadth NaN."""
    closes = _sector_closes(
        n=40, return_signs={s: +1 for s in SECTOR_ETFS}
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    lookback = v2_breadth_config.sector_breadth_lookback_days
    assert out.sector_breadth.iloc[:lookback].isna().all()
    assert not np.isnan(out.sector_breadth.iloc[lookback])


def test_sector_breadth_missing_sector_yields_all_nan(v2_breadth_config):
    """Ambiguity Log entry #27: missing a single sector → entire output is NaN
    (we do NOT rebase the denominator for the strict feature). The separate
    available-sector proxy still computes using the present sectors."""
    closes = _sector_closes(
        n=60, return_signs={s: +1 for s in SECTOR_ETFS}
    )
    # Drop XLRE (the youngest US sector ETF, often absent before 2015).
    del closes["XLRE"]
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    assert out.sector_breadth.isna().all()
    valid = out.available_sector_breadth.dropna()
    assert len(valid) > 0
    np.testing.assert_allclose(valid.to_numpy(), 1.0, atol=1e-12)
    assert out.available_sector_count.iloc[-1] == 10
    assert out.missing_sector_count.iloc[-1] == 1
    assert out.missing_sector_symbols.iloc[-1] == "XLRE"


def test_sector_breadth_aligns_to_input_index(v2_breadth_config):
    closes = _sector_closes(
        n=60, return_signs={s: +1 for s in SECTOR_ETFS}
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    assert isinstance(out, BreadthV2Features)
    assert (out.sector_breadth.index == closes["XLB"].index).all()
    frame = out.to_frame()
    assert list(frame.columns) == [
        "sector_breadth",
        "available_sector_breadth",
        "available_sector_count",
        "missing_sector_count",
        "missing_sector_symbols",
    ]


# =============================================================================
# Feature-store + timeline integration (AGENTS rule A wire-first)
# =============================================================================


_INTEGRATION_AS_OF = date(2023, 12, 14)


def _sector_etf_closes_aligned_to_spy(spy_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Synthetic per-sector close series aligned to the SPY index. Uses a
    deterministic-but-varied multiplier per sector so each sector has a
    distinct return path (no pathological all-tied series)."""
    closes: dict[str, pd.Series] = {}
    for i, symbol in enumerate(SECTOR_ETFS):
        # Each sector ramps at slightly different rates → mixed sign of 21d return.
        rate = 0.0005 + i * 0.00005
        arr = 100.0 * np.exp(np.arange(len(spy_index)) * rate)
        closes[symbol] = pd.Series(arr, index=spy_index, name=symbol)
    return closes


def test_build_feature_store_populates_breadth_state_v2(
    market_df_for_asof, v2_breadth_config
):
    cfg = load_default_regime_config()
    market_df = market_df_for_asof(_INTEGRATION_AS_OF)
    # First build a context without sector closes to discover the SPY index.
    bootstrap_context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df,
        config=cfg,
    )
    sector_closes = _sector_etf_closes_aligned_to_spy(bootstrap_context.spy_ohlcv.index)
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df,
        config=cfg,
        sector_etf_closes=sector_closes,
    )
    store = build_feature_store(
        context, breadth_state_v2_config=v2_breadth_config
    )
    assert store.breadth_state_v2 is not None
    assert isinstance(store.breadth_state_v2, BreadthV2Features)
    assert (store.breadth_state_v2.sector_breadth.index == store.spy_index).all()
    # At least one non-NaN value past the lookback.
    assert store.breadth_state_v2.sector_breadth.dropna().shape[0] > 0


def test_build_feature_store_graceful_when_sector_data_absent(
    market_df_for_asof, v2_breadth_config
):
    """V2 config supplied but sector_etf_closes absent → breadth_state_v2 is None."""
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    store = build_feature_store(
        context, breadth_state_v2_config=v2_breadth_config
    )
    assert store.breadth_state_v2 is None


def test_build_feature_store_none_when_v2_config_absent(market_df_for_asof):
    """No v2 breadth config → breadth_state_v2 is None even if sector data present."""
    cfg = load_default_regime_config()
    bootstrap = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    sector_closes = _sector_etf_closes_aligned_to_spy(bootstrap.spy_ohlcv.index)
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
        sector_etf_closes=sector_closes,
    )
    store = build_feature_store(context)
    assert store.breadth_state_v2 is None


def test_v1_config_path_leaves_breadth_state_v2_none(market_df_for_asof):
    """V1 contract preservation: a v1-only config (no v2 sub-blocks) yields a
    feature store where breadth_state_v2 is None and the timeline builds."""
    cfg = load_default_regime_config()
    cfg_v1 = cfg.model_copy(update={"breadth_state_v2": None})
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
    assert store.breadth_state_v2 is None


def test_timeline_threads_breadth_state_v2_config(market_df_for_asof):
    """End-to-end wire test (AGENTS rule A): build_regime_timeline must accept
    a v2 breadth config and surface breadth_state_v2 features through the
    feature_store path."""
    engine = RegimeEngine()
    cfg = engine.config
    assert cfg.breadth_state_v2 is not None

    from regime_detection.market_context import (
        slice_context_to_recent_sessions,
    )
    from regime_detection.timeline import ENGINE_MINIMUM_HISTORY

    bootstrap = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    sector_closes = _sector_etf_closes_aligned_to_spy(bootstrap.spy_ohlcv.index)
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
        sector_etf_closes=sector_closes,
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
        breadth_state_v2_config=cfg.breadth_state_v2,
    )
    assert store.breadth_state_v2 is not None
    assert len(store.breadth_state_v2.sector_breadth) == len(store.spy_index)

    # build_regime_timeline must propagate the v2 config without raising
    # (slice 2.3 ships compute + seam only; no new outputs on RegimeTimeline).
    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg
    )
    assert len(timeline.outputs) == 5
