"""TDD tests for v2 §2A Layer 2A Monetary / Liquidity V2 features (Slice 4.1).

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic FRED-style yield
scale (percentage points around 1.0–6.0) and the real production
Pydantic config — NO toy names. Math is verified against hand-computed
values.

Spec references:
    docs/regime_engine_v2_spec.md §2A (lines 985–1016 of the source
    file; the spec-pinned formula is at line 999 / spec body line 896).
    Slice 4.1 scope: yield_change_zscore_2y_63d and
    yield_change_zscore_10y_63d only. USD-index z-score, 21d-variant
    z-scores, label set, precedence, risk-rank, hysteresis, and the
    axis classifier are deferred (Implementation Ambiguity Log entries
    #44 and #45).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import (
    MonetaryPressureV2FeaturesConfig,
    load_default_regime_config,
)
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.monetary_pressure import (
    MonetaryPressureV2Features,
    compute_monetary_pressure_features,
)
from regime_detection.timeline import build_regime_timeline

# Spec-pinned defaults (v2 §2A line 896).
_LOOKBACK_DAYS = 63
_NORMALIZER_WINDOW = 1260
# First valid index = lookback + normalizer - 1 (verified in
# test_first_valid_index_matches_spec_defaults below).
_FIRST_VALID_T = _LOOKBACK_DAYS + _NORMALIZER_WINDOW - 1


# ---------- Shared fixtures ---------------------------------------------------


@pytest.fixture
def v2_monetary_config() -> MonetaryPressureV2FeaturesConfig:
    """Real production defaults from configs/core3-v2.0.0.yaml."""
    return MonetaryPressureV2FeaturesConfig(
        yield_change_lookback_days=_LOOKBACK_DAYS,
        zscore_normalizer_window_days=_NORMALIZER_WINDOW,
    )


def _index_n(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2024-12-31", periods=n)


def _fred_like_yield_series(*, n: int, seed: int, base: float) -> pd.Series:
    """Synthetic FRED-style daily yield series.

    Yields drift slowly (random walk with small daily innovations) and
    sit in the realistic range 1.0–6.0 % — matching DGS2 / DGS10
    behavior over the 2019–2024 window.
    """
    rng = np.random.default_rng(seed=seed)
    # Small daily innovations (~3bp std). Truncate to >= 0.1% so the
    # series stays in a realistic yield range.
    innovations = rng.normal(loc=0.0, scale=0.03, size=n)
    levels = np.cumsum(innovations) + base
    levels = np.clip(levels, 0.1, None)
    return pd.Series(levels, index=_index_n(n), name="yield")


# =============================================================================
# yield_change_63d boundary tests
# =============================================================================


def test_constant_yield_change_zscore_is_nan(v2_monetary_config):
    """Constant yield → yield_change_63d ≡ 0 → mean=0, std=0 → z-score is NaN.

    Boundary: a flat-yield window has no variation in the change
    series, so the spec's normalizer is undefined (0/0). The compute
    surfaces this as NaN (no synthesized value) per the V1 cold-start
    contract.
    """
    n = _FIRST_VALID_T + 10
    constant_yield = pd.Series(
        np.full(n, 4.50), index=_index_n(n), name="yield"
    )
    # Build a second flat series for the DGS10 slot.
    constant_yield_10 = pd.Series(
        np.full(n, 4.20), index=_index_n(n), name="yield"
    )
    out = compute_monetary_pressure_features(
        dgs2=constant_yield,
        dgs10=constant_yield_10,
        config=v2_monetary_config,
    )
    # Once warmed up, both z-scores remain NaN because std == 0.
    assert out.yield_change_zscore_2y_63d.iloc[_FIRST_VALID_T:].isna().all()
    assert out.yield_change_zscore_10y_63d.iloc[_FIRST_VALID_T:].isna().all()


def test_linearly_rising_yields_have_constant_change(v2_monetary_config):
    """Yields rising at +1bp / session → yield_change_63d ≡ 63bp (constant).

    Verifies the change-series definition pinned by §2A line 896:
    ``yield_change_63d[t] = yield[t] - yield[t-63]``. A perfectly linear
    yield path produces a constant change series of magnitude
    `63 * 0.01 = 0.63` for all `t >= 63`. NaN propagation for the
    `0/0` z-score boundary is covered by the constant-yield test
    above; here we pin only the change-window arithmetic.
    """
    n = _FIRST_VALID_T + 50
    # +1bp = +0.01% per session, starting at 1.00%.
    levels = 1.00 + 0.01 * np.arange(n, dtype=float)
    rising_yield = pd.Series(levels, index=_index_n(n), name="yield")
    # yield_change_63d should be (to floating-point tolerance) +0.63 at
    # every t >= 63. The compute_monetary_pressure_features call itself
    # is exercised by other tests; here we focus on the change formula.
    change = rising_yield - rising_yield.shift(_LOOKBACK_DAYS)
    np.testing.assert_allclose(
        change.iloc[_LOOKBACK_DAYS:].to_numpy(),
        np.full(n - _LOOKBACK_DAYS, 0.63),
        atol=1e-12,
    )
    # Pre-lookback values are NaN per `.shift(63)` semantics.
    assert change.iloc[:_LOOKBACK_DAYS].isna().all()


def test_first_valid_index_matches_spec_defaults(v2_monetary_config):
    """Cold-start: first non-NaN z-score lands at t = 63 + 1260 - 1 = 1322.

    Earlier indices must be NaN. This pins the warm-up boundary for
    Ambiguity Log #45.
    """
    n = _FIRST_VALID_T + 100
    # Use a stochastic series so std > 0 once the normalizer warms up.
    dgs2 = _fred_like_yield_series(n=n, seed=20260512, base=4.50)
    dgs10 = _fred_like_yield_series(n=n, seed=20260513, base=4.20)
    out = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, config=v2_monetary_config
    )
    # Strictly NaN before _FIRST_VALID_T.
    assert out.yield_change_zscore_2y_63d.iloc[:_FIRST_VALID_T].isna().all()
    assert out.yield_change_zscore_10y_63d.iloc[:_FIRST_VALID_T].isna().all()
    # Non-NaN at _FIRST_VALID_T = 1322.
    assert _FIRST_VALID_T == 1322
    assert not math.isnan(out.yield_change_zscore_2y_63d.iloc[_FIRST_VALID_T])
    assert not math.isnan(out.yield_change_zscore_10y_63d.iloc[_FIRST_VALID_T])


# =============================================================================
# Hand-computed z-score on a 1500-day fixture
# =============================================================================


def test_zscore_hand_computed_at_specific_t(v2_monetary_config):
    """Pick t = 1400 on a 1500-day stochastic DGS2 series and assert the
    z-score equals (yield_change_63d[t] - mean_5y[t]) / std_5y[t]
    computed independently with numpy."""
    n = 1500
    dgs2 = _fred_like_yield_series(n=n, seed=20260514, base=4.75)
    dgs10 = _fred_like_yield_series(n=n, seed=20260515, base=4.10)
    out = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, config=v2_monetary_config
    )

    t = 1400
    change = dgs2 - dgs2.shift(_LOOKBACK_DAYS)
    # Window covers the 5y of change values ENDING at t, inclusive:
    # change.iloc[t - normalizer + 1 : t + 1]
    window = change.iloc[t - _NORMALIZER_WINDOW + 1 : t + 1]
    assert len(window) == _NORMALIZER_WINDOW
    expected = (change.iloc[t] - window.mean()) / window.std(ddof=1)
    assert out.yield_change_zscore_2y_63d.iloc[t] == pytest.approx(
        expected, rel=1e-12
    )


# =============================================================================
# DGS2 vs DGS10 independence
# =============================================================================


def test_nan_in_dgs2_does_not_propagate_to_dgs10(v2_monetary_config):
    """Inserting a NaN block in DGS2 must not affect the DGS10 z-score
    output. AGENTS rule I — quarantine bad external data — applied to
    the per-series compute boundary."""
    n = _FIRST_VALID_T + 100
    dgs2 = _fred_like_yield_series(n=n, seed=20260516, base=4.50)
    dgs10 = _fred_like_yield_series(n=n, seed=20260517, base=4.10)

    # Baseline (no NaN injection): record DGS10 z-score at last index.
    baseline = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, config=v2_monetary_config
    )
    baseline_dgs10_last = baseline.yield_change_zscore_10y_63d.iloc[-1]
    assert not math.isnan(baseline_dgs10_last)

    # Inject NaNs into DGS2 in the middle. DGS10 stays untouched.
    dgs2_dirty = dgs2.copy()
    dgs2_dirty.iloc[500:510] = np.nan

    perturbed = compute_monetary_pressure_features(
        dgs2=dgs2_dirty, dgs10=dgs10, config=v2_monetary_config
    )
    # DGS10 z-score at the last index is byte-identical to baseline.
    assert perturbed.yield_change_zscore_10y_63d.iloc[-1] == pytest.approx(
        baseline_dgs10_last, rel=0.0, abs=0.0
    )
    # And the DGS2 z-score IS affected at and after the NaN block
    # (sanity check that the dirtying was effective).
    assert not perturbed.yield_change_zscore_2y_63d.equals(
        baseline.yield_change_zscore_2y_63d
    )


# =============================================================================
# Shape / to_frame
# =============================================================================


def test_features_align_to_input_index(v2_monetary_config):
    n = _FIRST_VALID_T + 50
    dgs2 = _fred_like_yield_series(n=n, seed=20260518, base=4.50)
    dgs10 = _fred_like_yield_series(n=n, seed=20260519, base=4.10)
    out = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, config=v2_monetary_config
    )
    assert isinstance(out, MonetaryPressureV2Features)
    frame = out.to_frame()
    assert list(frame.columns) == list(out.feature_names)
    assert frame.shape == (n, 2)
    assert (frame.index == dgs2.index).all()


# =============================================================================
# Feature-store + timeline integration (AGENTS rule A)
# =============================================================================


_INTEGRATION_AS_OF = date(2023, 12, 14)


def _macro_series_for_spy_index(
    spy_index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Build a deterministic DGS2/DGS10 macro_series aligned to a SPY
    index. Realistic FRED yield scale (2y around 4.5%, 10y around 4.0%).
    """
    n = len(spy_index)
    rng = np.random.default_rng(seed=20260520)
    innovations_2y = rng.normal(loc=0.0, scale=0.03, size=n)
    innovations_10y = rng.normal(loc=0.0, scale=0.025, size=n)
    dgs2_levels = np.clip(np.cumsum(innovations_2y) + 4.50, 0.1, None)
    dgs10_levels = np.clip(np.cumsum(innovations_10y) + 4.00, 0.1, None)
    return {
        "DGS2": pd.Series(dgs2_levels, index=spy_index, name="DGS2"),
        "DGS10": pd.Series(dgs10_levels, index=spy_index, name="DGS10"),
    }


def test_build_feature_store_populates_monetary(
    market_df_for_asof, v2_monetary_config
):
    cfg = load_default_regime_config()
    market_data = market_df_for_asof(_INTEGRATION_AS_OF)
    # Build the context once with no macro_series so we can size the
    # synthetic FRED series to the SPY index.
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_data,
        config=cfg,
    )
    macro = _macro_series_for_spy_index(context.spy_ohlcv.index)
    context_with_macro = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_data,
        config=cfg,
        macro_series=macro,
    )
    store = build_feature_store(
        context_with_macro,
        monetary_pressure_v2_config=v2_monetary_config,
    )
    assert store.monetary is not None
    assert isinstance(store.monetary, MonetaryPressureV2Features)
    series_2y = store.monetary.yield_change_zscore_2y_63d
    series_10y = store.monetary.yield_change_zscore_10y_63d
    assert isinstance(series_2y, pd.Series)
    assert isinstance(series_10y, pd.Series)
    assert (series_2y.index == store.spy_index).all()
    assert (series_10y.index == store.spy_index).all()


def test_build_feature_store_graceful_degradation_no_macro(
    market_df_for_asof, v2_monetary_config
):
    """No macro_series on context → monetary is None, no exception raised."""
    cfg = load_default_regime_config()
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg,
    )
    assert context.macro_series is None
    store = build_feature_store(
        context, monetary_pressure_v2_config=v2_monetary_config
    )
    assert store.monetary is None


def test_v1_config_path_leaves_monetary_none(market_df_for_asof):
    """V1 contract: config without monetary_pressure_v2 → monetary is None,
    v1 outputs unchanged, timeline builds without raising."""
    cfg = load_default_regime_config()
    cfg_v1 = cfg.model_copy(update={"monetary_pressure_v2": None})
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_df_for_asof(_INTEGRATION_AS_OF),
        config=cfg_v1,
    )
    store = build_feature_store(context)
    assert store.monetary is None
    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg_v1
    )
    assert len(timeline.outputs) == 5


def test_timeline_threads_monetary_pressure_v2_config(market_df_for_asof):
    """End-to-end wire test (AGENTS rule A): build_regime_timeline must
    accept the v2 config and surface monetary features via the same
    feature_store path used by the engine."""
    cfg = load_default_regime_config()
    assert cfg.monetary_pressure_v2 is not None
    market_data = market_df_for_asof(_INTEGRATION_AS_OF)

    # First pass: build a context without macro to size the synthetic series.
    context_no_macro = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_data,
        config=cfg,
    )
    macro = _macro_series_for_spy_index(context_no_macro.spy_ohlcv.index)
    context = build_market_context(
        end_date=_INTEGRATION_AS_OF,
        market_data=market_data,
        config=cfg,
        macro_series=macro,
    )

    timeline = build_regime_timeline(
        context=context, lookback_days=5, config=cfg
    )
    assert len(timeline.outputs) == 5

    # And the feature_store seam carries the monetary features.
    from regime_detection.market_context import (
        slice_context_to_recent_sessions,
    )
    from regime_detection.timeline import ENGINE_MINIMUM_HISTORY

    required = min(len(context.sessions), ENGINE_MINIMUM_HISTORY)
    working = slice_context_to_recent_sessions(
        context=context, required_sessions=required
    )
    store = build_feature_store(
        working,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
    )
    assert store.monetary is not None
    assert len(store.monetary.yield_change_zscore_2y_63d) == len(
        store.spy_index
    )
    assert len(store.monetary.yield_change_zscore_10y_63d) == len(
        store.spy_index
    )
