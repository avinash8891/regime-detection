from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection.breadth_state import BreadthFeatures, compute_features as compute_breadth_features
from regime_detection.breadth_state_v2 import (
    BreadthV2Features,
    compute_breadth_v2_features,
)
from regime_detection.config import (
    BreadthV2Config,
    NetworkFragilityConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
)
from regime_detection.fragility_universe import SECTOR_ETFS
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
from regime_detection.trend_direction_v2 import (
    TrendDirectionV2Features,
    compute_trend_v2_features,
)
from regime_detection.volatility_state import VolatilityFeatures, compute_features as compute_volatility_features
from regime_detection.volatility_state_v2 import (
    VolatilityV2Features,
    compute_volatility_v2_features,
)

__all__ = [
    "BreadthV2Features",
    "FeatureStore",
    "NetworkFragilityFeatures",
    "TrendDirectionV2Features",
    "VolatilityV2Features",
    "build_feature_store",
]


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

    # V2 §1A seam — populated when a TrendDirectionV2Config is threaded
    # through. SPY close is always present on the V1+V2 path so this is
    # only None when the config is absent (v1-only callers).
    trend_direction_v2: TrendDirectionV2Features | None = None

    # V2 §1C seam — populated when a VolatilityV2Config is threaded through.
    # SPY OHLC is always present on the V1+V2 path so this is only None when
    # the config is absent (v1-only callers).
    volatility_state_v2: VolatilityV2Features | None = None

    # V2 §1D seam — populated when a BreadthV2Config is threaded through AND
    # context.sector_etf_closes is non-None with all 11 sector symbols
    # present. Otherwise None (graceful degradation — V2 §1D PIT pipeline is
    # not yet ingested for related features; sector ETF feed is optional).
    breadth_state_v2: BreadthV2Features | None = None


def build_feature_store(
    context: MarketContext,
    *,
    network_fragility_config: NetworkFragilityConfig | None = None,
    trend_direction_v2_config: TrendDirectionV2Config | None = None,
    volatility_state_v2_config: VolatilityV2Config | None = None,
    breadth_state_v2_config: BreadthV2Config | None = None,
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

    # V2 §1A trend-direction features (slice 2.1) — evidence-only compute.
    if trend_direction_v2_config is not None:
        trend_direction_v2 = compute_trend_v2_features(
            spy_close, config=trend_direction_v2_config
        )
    else:
        trend_direction_v2 = None

    # V2 §1C volatility features (slice 2.2) — evidence-only compute.
    if volatility_state_v2_config is not None:
        volatility_state_v2 = compute_volatility_v2_features(
            open_=spy_ohlcv["open"],
            high=spy_ohlcv["high"],
            low=spy_ohlcv["low"],
            close=spy_close,
            config=volatility_state_v2_config,
        )
    else:
        volatility_state_v2 = None

    # V2 §1D breadth features (slice 2.3) — evidence-only compute. Requires
    # all 11 sector ETFs in MarketContext.sector_etf_closes (Ambiguity Log
    # entry #27 pins the missing-sector policy as fail-NaN). When the config
    # is supplied but the data is missing or partial, fall back to None
    # (matches the slice 1.2 NetworkFragility seam pattern).
    if breadth_state_v2_config is not None and context.sector_etf_closes is not None:
        sector_closes = context.sector_etf_closes
        if all(symbol in sector_closes for symbol in SECTOR_ETFS):
            breadth_state_v2 = compute_breadth_v2_features(
                sector_etf_closes=sector_closes,
                config=breadth_state_v2_config,
            )
        else:
            breadth_state_v2 = None
    else:
        breadth_state_v2 = None

    return FeatureStore(
        spy_index=spy_ohlcv.index,
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility=volatility,
        breadth=breadth,
        sma_50=sma_50,
        network_fragility=network_fragility,
        trend_direction_v2=trend_direction_v2,
        volatility_state_v2=volatility_state_v2,
        breadth_state_v2=breadth_state_v2,
    )
