"""V2 Slice 7 — failing tests for the GMM clustering evidence layer.

Spec pins: ``docs/regime_engine_v2_spec.md`` §6.2 (lines 2808-2858).
The clustering module reuses existing FeatureStore seams as inputs and
emits raw per-day cluster IDs + probabilities + Mahalanobis distance to
the assigned-cluster centroid. NEVER auto-maps cluster IDs to economic
labels (operator-side mapping per V2 §10 + spec line 2837).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import (
    ClusteringConfig,
    load_default_regime_config,
)
from regime_detection.clustering import (
    ClusteringFeatures,
    compute_clustering_features,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic but deterministic inputs that mimic the 7 spec seams.
# ---------------------------------------------------------------------------


def _synthetic_inputs(
    n_sessions: int = 1500, *, seed: int = 0
) -> dict[str, pd.Series]:
    """Build seven synthetic series with two visible regimes."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n_sessions)
    half = n_sessions // 2

    calm_ret = rng.normal(loc=0.0005, scale=0.005, size=half)
    vol_ret = rng.normal(loc=-0.0005, scale=0.020, size=n_sessions - half)
    returns = np.concatenate([calm_ret, vol_ret])
    return_1d = pd.Series(returns, index=index)

    price = (1.0 + return_1d).cumprod() * 100.0

    return_21d = (price / price.shift(21) - 1.0).rename("return_21d")
    return_63d = (price / price.shift(63) - 1.0).rename("return_63d")

    realized_vol_21d = (
        return_1d.rolling(21).std() * np.sqrt(252)
    ).rename("realized_vol_21d")

    peak_63 = price.rolling(63, min_periods=63).max()
    drawdown_63d = (price / peak_63 - 1.0).rename("drawdown_63d")

    adx_calm = rng.normal(loc=18.0, scale=3.0, size=half).clip(0.0, 100.0)
    adx_vol = rng.normal(loc=35.0, scale=5.0, size=n_sessions - half).clip(
        0.0, 100.0
    )
    adx_14 = pd.Series(
        np.concatenate([adx_calm, adx_vol]), index=index, name="adx_14"
    )

    corr_calm = rng.normal(loc=0.30, scale=0.05, size=half).clip(0.0, 0.95)
    corr_vol = rng.normal(loc=0.65, scale=0.05, size=n_sessions - half).clip(
        0.0, 0.95
    )
    avg_pairwise_corr_63d = pd.Series(
        np.concatenate([corr_calm, corr_vol]),
        index=index,
        name="avg_pairwise_corr_63d",
    )

    pct_calm = rng.normal(loc=0.70, scale=0.07, size=half).clip(0.0, 1.0)
    pct_vol = rng.normal(loc=0.30, scale=0.10, size=n_sessions - half).clip(
        0.0, 1.0
    )
    pct_above_50dma = pd.Series(
        np.concatenate([pct_calm, pct_vol]),
        index=index,
        name="pct_above_50dma",
    )

    return {
        "return_21d": return_21d,
        "return_63d": return_63d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "adx_14": adx_14,
        "avg_pairwise_corr_63d": avg_pairwise_corr_63d,
        "pct_above_50dma": pct_above_50dma,
    }


def _default_clustering_config(
    training_window_days: int = 1260,
) -> ClusteringConfig:
    return ClusteringConfig(
        n_clusters=8,
        training_window_days=training_window_days,
        random_state=42,
        covariance_type="full",
        model_version="gmm_8cluster_v1.0",
    )


@pytest.fixture(scope="module")
def _computed_default_clustering_pair() -> tuple[ClusteringFeatures, ClusteringFeatures]:
    inputs = _synthetic_inputs(n_sessions=1500)
    cfg = _default_clustering_config()
    first = compute_clustering_features(config=cfg, **inputs)
    second = compute_clustering_features(config=cfg, **inputs)
    assert first is not None and second is not None
    return first, second


@pytest.fixture(scope="module")
def _computed_default_clustering(
    _computed_default_clustering_pair: tuple[ClusteringFeatures, ClusteringFeatures],
) -> ClusteringFeatures:
    return _computed_default_clustering_pair[0]


# ---------------------------------------------------------------------------
# Group A — compute_clustering_features unit tests
# ---------------------------------------------------------------------------


def test_compute_clustering_features_returns_none_when_any_input_is_none() -> None:
    inputs = _synthetic_inputs()
    cfg = _default_clustering_config()
    for missing in inputs:
        call_kwargs = {k: (None if k == missing else v) for k, v in inputs.items()}
        result = compute_clustering_features(config=cfg, **call_kwargs)
        assert result is None, f"missing {missing} → expected None"


def test_compute_clustering_features_returns_none_when_insufficient_history() -> None:
    inputs = _synthetic_inputs(n_sessions=100)
    cfg = _default_clustering_config(training_window_days=1260)
    result = compute_clustering_features(config=cfg, **inputs)
    assert result is None


def test_compute_clustering_features_succeeds_on_synthetic_inputs(
    _computed_default_clustering: ClusteringFeatures,
) -> None:
    inputs = _synthetic_inputs(n_sessions=1500)
    result = _computed_default_clustering
    assert result is not None
    assert isinstance(result, ClusteringFeatures)
    assert result.n_clusters == 8
    assert result.model_version == "gmm_8cluster_v1.0"
    assert len(result.cluster_id) == 1500
    non_null = result.cluster_id.dropna()
    # V1 §2.2 PIT semantics: every warmed session gets a cluster assignment
    # from a GMM trained only on data available through that session.
    assert len(non_null) > 30
    assert non_null.index.max() == inputs["return_21d"].dropna().index[-1]
    for val in non_null.unique():
        assert int(val) in range(8)


def test_cluster_probabilities_sum_to_one_per_session(
    _computed_default_clustering: ClusteringFeatures,
) -> None:
    result = _computed_default_clustering
    assert result is not None
    valid_rows = result.cluster_probabilities.dropna(how="any")
    assert len(valid_rows) > 0
    row_sums = valid_rows.sum(axis=1)
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-6)


def test_cluster_id_is_argmax_of_probabilities(
    _computed_default_clustering: ClusteringFeatures,
) -> None:
    result = _computed_default_clustering
    assert result is not None
    valid_proba = result.cluster_probabilities.dropna(how="any")
    valid_id = result.cluster_id.dropna()
    common_idx = valid_proba.index.intersection(valid_id.index)
    assert len(common_idx) > 0
    sample_ts = common_idx[len(common_idx) // 2]
    expected = int(valid_proba.loc[sample_ts].idxmax())
    actual = int(valid_id.loc[sample_ts])
    assert actual == expected


def test_distance_to_centroid_is_non_negative(
    _computed_default_clustering: ClusteringFeatures,
) -> None:
    result = _computed_default_clustering
    assert result is not None
    non_null = result.distance_to_centroid.dropna()
    assert len(non_null) > 0
    assert (non_null >= 0.0).all()


def test_seed_determinism(
    _computed_default_clustering_pair: tuple[ClusteringFeatures, ClusteringFeatures],
) -> None:
    first, second = _computed_default_clustering_pair
    assert first is not None and second is not None
    pd.testing.assert_series_equal(first.cluster_id, second.cluster_id)
    pd.testing.assert_series_equal(
        first.distance_to_centroid, second.distance_to_centroid
    )


def test_compute_clustering_returns_none_on_singular_covariance() -> None:
    """Constant zero-variance inputs force a singular covariance failure."""
    index = pd.bdate_range("2010-01-04", periods=1500)
    constant_series = pd.Series(np.zeros(1500), index=index)
    inputs = {
        "return_21d": constant_series.copy(),
        "return_63d": constant_series.copy(),
        "realized_vol_21d": constant_series.copy(),
        "drawdown_63d": constant_series.copy(),
        "adx_14": constant_series.copy(),
        "avg_pairwise_corr_63d": constant_series.copy(),
        "pct_above_50dma": constant_series.copy(),
    }
    cfg = _default_clustering_config()
    result = compute_clustering_features(config=cfg, **inputs)
    assert result is None


# ---------------------------------------------------------------------------
# Group B — FeatureStore seam wiring
# ---------------------------------------------------------------------------


def test_feature_store_clustering_seam_none_when_config_absent(
    raw_market_frames: dict[str, pd.DataFrame],
) -> None:
    from regime_detection.config import load_default_regime_config
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context
    from regime_detection.calendar import require_nyse_trading_day

    cfg = load_default_regime_config().model_copy(update={"clustering": None})
    spy = raw_market_frames["SPY"]
    rsp = raw_market_frames["RSP"]
    vixy = raw_market_frames["VIXY"]
    raw = pd.concat([spy, rsp, vixy], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    last_session = max(d for d in raw["date"].unique())
    while True:
        try:
            require_nyse_trading_day(last_session)
            break
        except Exception:
            last_session = last_session.fromordinal(last_session.toordinal() - 1)
    market_data = raw[raw["date"] <= last_session].copy().reset_index(drop=True)
    context = build_market_context(
        end_date=last_session,
        market_data=market_data,
        config=cfg,
    )
    feature_store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
        volatility_state_v2_config=cfg.volatility_state_v2,
        breadth_state_v2_config=cfg.breadth_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
    )
    assert feature_store.clustering is None


def test_feature_store_clustering_seam_none_when_pct_above_50dma_absent(
    raw_market_frames: dict[str, pd.DataFrame],
) -> None:
    """Without PIT inputs, pct_above_50dma stays None → clustering seam None."""
    from regime_detection.config import load_default_regime_config
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context
    from regime_detection.calendar import require_nyse_trading_day

    cfg = load_default_regime_config()  # has clustering, no PIT inputs
    assert cfg.clustering is not None
    spy = raw_market_frames["SPY"]
    rsp = raw_market_frames["RSP"]
    vixy = raw_market_frames["VIXY"]
    raw = pd.concat([spy, rsp, vixy], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    last_session = max(d for d in raw["date"].unique())
    while True:
        try:
            require_nyse_trading_day(last_session)
            break
        except Exception:
            last_session = last_session.fromordinal(last_session.toordinal() - 1)
    market_data = raw[raw["date"] <= last_session].copy().reset_index(drop=True)
    context = build_market_context(
        end_date=last_session,
        market_data=market_data,
        config=cfg,
    )
    feature_store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        trend_direction_v2_config=cfg.trend_direction_v2,
        volatility_state_v2_config=cfg.volatility_state_v2,
        breadth_state_v2_config=cfg.breadth_state_v2,
        volume_liquidity_v2_config=cfg.volume_liquidity_v2,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
    )
    # No PIT inputs → breadth_state_v2 may exist but pct_above_50dma is None,
    # or breadth_state_v2 itself is None. Either way clustering seam is None.
    assert feature_store.clustering is None


# ---------------------------------------------------------------------------
# Group C — Integration tests
# ---------------------------------------------------------------------------


def test_real_default_config_carries_clustering_block() -> None:
    cfg = load_default_regime_config()
    assert cfg.clustering is not None
    assert cfg.clustering.n_clusters == 8
    assert cfg.clustering.model_version == "gmm_8cluster_v1.0"
    assert cfg.clustering.training_window_days == 1260
    assert cfg.clustering.random_state == 42


def test_regime_output_omits_cluster_field_when_clustering_seam_none(
    raw_market_data: pd.DataFrame,
    market_df_for_asof,
) -> None:
    """Default classify() (no PIT inputs) → clustering seam None → RegimeOutput.cluster None and omitted from JSON dump."""
    from regime_detection.engine import RegimeEngine

    engine = RegimeEngine()
    last_session = max(raw_market_data["date"].unique())
    market_data = market_df_for_asof(last_session)
    out = engine.classify(as_of_date=last_session, market_data=market_data)
    assert out.cluster is None
    dumped = out.model_dump()
    assert "cluster" not in dumped
