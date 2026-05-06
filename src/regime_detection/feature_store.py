from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection.breadth_state import BreadthFeatures, compute_features as compute_breadth_features
from regime_detection.market_context import MarketContext
from regime_detection.trend_character import (
    TrendCharacterFeatures,
    compute_features as compute_trend_character_features,
)
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    compute_features as compute_trend_direction_features,
)
from regime_detection.volatility_state import VolatilityFeatures, compute_features as compute_volatility_features


class FeatureStore(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spy_index: pd.DatetimeIndex
    trend_direction: TrendDirectionFeatures
    trend_character: TrendCharacterFeatures
    volatility: VolatilityFeatures
    breadth: BreadthFeatures
    sma_50: pd.Series


def build_feature_store(context: MarketContext) -> FeatureStore:
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
    return FeatureStore(
        spy_index=spy_ohlcv.index,
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility=volatility,
        breadth=breadth,
        sma_50=sma_50,
    )
