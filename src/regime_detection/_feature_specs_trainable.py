"""Trainable-evidence feature spec resolvers for the feature store."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from regime_detection.breadth_state import BreadthV2Features
from regime_detection.feature_store_runtime import (
    _Unavailable,
    _require_build_input,
    _require_feature,
)
from regime_detection.market_context import MarketContext
from regime_detection.network_fragility import NetworkFragilityFeatures
from regime_detection.trend_character import TrendCharacterFeatures
from regime_detection.trend_direction import TrendDirectionV2Features
from regime_detection.volatility_state import VolatilityFeatures
from regime_detection.volume_liquidity import VolumeLiquidityV2Features


class TrainableFeatureState(Protocol):
    context: MarketContext
    spy_close: pd.Series
    volatility: VolatilityFeatures | None
    volume_liquidity_v2: VolumeLiquidityV2Features | None
    network_fragility: NetworkFragilityFeatures | None
    realized_vol_21d: pd.Series | None
    drawdown_63d: pd.Series | None
    trend_character: TrendCharacterFeatures | None
    trend_direction_v2: TrendDirectionV2Features | None
    breadth_state_v2: BreadthV2Features | None


def resolve_drawdown_63d(
    state: TrainableFeatureState,
) -> dict[str, object] | _Unavailable:
    """Build drawdown only when HMM or clustering consumes it."""
    if state.context.config.hmm is None and state.context.config.clustering is None:
        return _Unavailable(missing_inputs=("hmm_or_clustering_config",))
    return {"close": state.spy_close, "lookback": 63}


def resolve_hmm(
    state: TrainableFeatureState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.context.config.hmm is None:
        missing.append("hmm_config")
    if state.volume_liquidity_v2 is None:
        missing.append("volume_liquidity_v2")
    if state.network_fragility is None:
        missing.append("network_fragility")
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    volatility = _require_feature(state.volatility, "volatility")
    volume_liquidity_v2 = _require_feature(
        state.volume_liquidity_v2, "volume_liquidity_v2"
    )
    network_fragility = _require_feature(state.network_fragility, "network_fragility")
    realized_vol_21d = _require_feature(state.realized_vol_21d, "realized_vol_21d")
    drawdown_63d = _require_feature(state.drawdown_63d, "drawdown_63d")
    return {
        "config": state.context.config.hmm,
        "return_1d": volatility.return_1d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "volume_zscore_20d": volume_liquidity_v2.volume_zscore_20d,
        "avg_pairwise_corr_63d": network_fragility.avg_pairwise_corr_63d,
    }


def resolve_clustering(
    state: TrainableFeatureState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.context.config.clustering is None:
        missing.append("clustering_config")
    breadth_state_v2 = state.breadth_state_v2
    if breadth_state_v2 is None or breadth_state_v2.pct_above_50dma is None:
        missing.append("breadth_state_v2.pct_above_50dma")
    if state.network_fragility is None:
        missing.append("network_fragility")
    if state.trend_direction_v2 is None:
        missing.append("trend_direction_v2")
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    trend_character = _require_feature(state.trend_character, "trend_character")
    trend_direction_v2 = _require_feature(
        state.trend_direction_v2, "trend_direction_v2"
    )
    network_fragility = _require_feature(state.network_fragility, "network_fragility")
    realized_vol_21d = _require_feature(state.realized_vol_21d, "realized_vol_21d")
    drawdown_63d = _require_feature(state.drawdown_63d, "drawdown_63d")
    breadth_state_v2 = _require_feature(breadth_state_v2, "breadth_state_v2")
    pct_above_50dma = _require_build_input(
        breadth_state_v2.pct_above_50dma, "breadth_state_v2.pct_above_50dma"
    )
    return {
        "config": state.context.config.clustering,
        "return_21d": trend_character.return_21d,
        "return_63d": trend_direction_v2.return_63d,
        "realized_vol_21d": realized_vol_21d,
        "drawdown_63d": drawdown_63d,
        "adx_14": trend_character.adx_14,
        "avg_pairwise_corr_63d": network_fragility.avg_pairwise_corr_63d,
        "pct_above_50dma": pct_above_50dma,
    }


def resolve_change_point(
    state: TrainableFeatureState,
) -> dict[str, object] | _Unavailable:
    if state.context.config.change_point is None:
        return _Unavailable(missing_inputs=("change_point_config",))
    if state.realized_vol_21d is None:
        return _Unavailable(missing_inputs=("realized_vol_21d",))
    return {
        "realized_vol_21d": state.realized_vol_21d,
        "config": state.context.config.change_point,
    }


def resolve_realized_vol_21d(
    state: TrainableFeatureState,
) -> dict[str, object] | _Unavailable:
    """Build realized vol only when trainable evidence consumes it."""
    if (
        state.context.config.hmm is None
        and state.context.config.clustering is None
        and state.context.config.change_point is None
    ):
        return _Unavailable(
            missing_inputs=("hmm_or_clustering_or_change_point_config",)
        )
    return {"close": state.spy_close, "window": 21}
