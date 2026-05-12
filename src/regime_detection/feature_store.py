from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection.breadth_state import BreadthFeatures, compute_features as compute_breadth_features
from regime_detection.config import NetworkFragilityConfig
from regime_detection.market_context import MarketContext
from regime_detection.network_fragility import (
    NetworkFragilityFeatures,
    compute_features as compute_network_fragility_features,
)
from regime_detection.trend_character import (
    TrendCharacterFeatures,
    compute_features as compute_trend_character_features,
)
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    compute_features as compute_trend_direction_features,
)
from regime_detection.volatility_state import VolatilityFeatures, compute_features as compute_volatility_features

__all__ = ["FeatureStore", "NetworkFragilityFeatures", "build_feature_store"]


class FeatureStore(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spy_index: pd.DatetimeIndex
    trend_direction: TrendDirectionFeatures
    trend_character: TrendCharacterFeatures
    volatility: VolatilityFeatures
    breadth: BreadthFeatures
    sma_50: pd.Series

    # V2 §3 seam — populated when context.sector_etf_closes is present.
    # Slice 1 swaps the placeholder for the real feature compute.
    network_fragility: NetworkFragilityFeatures | None = None


def build_feature_store(
    context: MarketContext,
    *,
    network_fragility_config: NetworkFragilityConfig | None = None,
) -> FeatureStore:
    spy_ohlcv = context.spy_ohlcv
    spy_close = spy_ohlcv["close"]
    trend_direction = compute_trend_direction_features(spy_close)
    trend_character = compute_trend_character_features(
        close=spy_close,
        high=spy_ohlcv["high"],
        low=spy_ohlcv["low"],
    )
    volatility = compute_volatility_features(
        close=spy_close,
        vix_proxy_close=context.vix_proxy_close,
    )
    breadth = compute_breadth_features(
        spy_close=spy_close,
        rsp_close=context.rsp_close.reindex(spy_ohlcv.index),
    )
    sma_50 = spy_close.rolling(50).mean()
    # V2 §3.2 feature compute (slice 1.2). The classifier wiring (slice 1.3+)
    # consumes these series; for now they populate the seam so build_feature_store
    # returns a typed NetworkFragilityFeatures whenever sector data is present.
    if context.sector_etf_closes is not None:
        nf_kwargs: dict[str, int | float] = {}
        if network_fragility_config is not None:
            nf_kwargs = {
                "correlation_lookback_days": network_fragility_config.correlation_lookback_days,
                "percentile_lookback_days": network_fragility_config.percentile_lookback_days,
                "realized_vol_lookback_days": network_fragility_config.realized_vol_lookback_days,
                "dispersion_percentile_lookback_days": (
                    network_fragility_config.dispersion_percentile_lookback_days
                ),
                "min_universe_size": network_fragility_config.min_universe_size,
                "min_window_completeness": network_fragility_config.min_window_completeness,
            }
        network_fragility = compute_network_fragility_features(
            sector_etf_closes=context.sector_etf_closes,
            cross_asset_closes=context.cross_asset_closes or {},
            spy_close=spy_close,
            **nf_kwargs,
        )
    else:
        network_fragility = None
    return FeatureStore(
        spy_index=spy_ohlcv.index,
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility=volatility,
        breadth=breadth,
        sma_50=sma_50,
        network_fragility=network_fragility,
    )
