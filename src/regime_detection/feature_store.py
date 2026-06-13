from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection.breadth_state import BreadthFeatures, BreadthV2Features
from regime_detection.change_point import ChangePointFeatures
from regime_detection.clustering import ClusteringFeatures
from regime_detection.config import (
    BreadthV2Config,
    CentralBankTextConfig,
    CreditFundingConfig,
    InflationGrowthConfig,
    MonetaryPressureV2FeaturesConfig,
    NetworkFragilityConfig,
    NewsSentimentConfig,
    SentimentScoreConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
    VolumeLiquidityV2Config,
)
from regime_detection.credit_funding import CreditFundingFeatures
from regime_detection.hmm_state import HMMFeatures
from regime_detection.inflation_growth import InflationGrowthFeatures
from regime_detection.market_context import MarketContext
from regime_detection.monetary_pressure import MonetaryPressureV2Features
from regime_detection.network_fragility import NetworkFragilityFeatures
from regime_detection.trend_character import TrendCharacterFeatures
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    TrendDirectionV2Features,
)
from regime_detection.volatility_state import VolatilityFeatures, VolatilityV2Features
from regime_detection.volume_liquidity import VolumeLiquidityV2Features
from regime_detection.feature_store_runtime import (
    FeatureAvailability,
    _run_feature_specs,
)
from regime_detection._feature_specs import (
    FEATURE_SPECS,
    FeatureStoreBuildState,
    as_datetime_index,
    require_feature,
    series_column,
)

_FeatureStoreBuildState = FeatureStoreBuildState
_FEATURE_SPECS = FEATURE_SPECS

__all__ = [
    "BreadthV2Features",
    "ChangePointFeatures",
    "ClusteringFeatures",
    "CreditFundingFeatures",
    "FeatureAvailability",
    "FeatureStore",
    "HMMFeatures",
    "InflationGrowthFeatures",
    "MonetaryPressureV2Features",
    "NetworkFragilityFeatures",
    "TrendDirectionV2Features",
    "VolatilityV2Features",
    "VolumeLiquidityV2Features",
    "build_feature_store",
]


class FeatureStore(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spy_index: pd.DatetimeIndex
    availability: dict[str, FeatureAvailability]
    trend_direction: TrendDirectionFeatures
    trend_character: TrendCharacterFeatures
    volatility: VolatilityFeatures
    breadth: BreadthFeatures
    sma_50: pd.Series

    # V2 §3 seam — populated when context.sector_etf_closes is present.
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
    # context.sector_etf_closes is non-None AND at least one symbol from
    # SECTOR_ETFS is present (any-of-1, not all-of-11). Authorization for
    # the any-of-1 gate lives in breadth_state_v2 module docstring L28-43:
    # the compute function emits two outputs — strict `sector_breadth_21d`
    # (NaN when any of 11 sectors missing, per §1D line 229 "divided by 11")
    # AND `available_sector_breadth_21d` proxy (uses available sectors,
    # exposes count + missing-symbol evidence). The proxy is computable
    # whenever at least one sector exists, so the seam admits any-of-1.
    # Otherwise None — V2 §1D PIT pipeline is not yet ingested for related
    # features; sector ETF feed is optional.
    breadth_state_v2: BreadthV2Features | None = None

    # V2 §1E seam — populated when a VolumeLiquidityV2Config is threaded
    # through AND a SPY volume series is available on the context. SPY
    # volume rides on MarketContext.spy_ohlcv["volume"] on the V1+V2 path
    # so this is only None when the config is absent (v1-only callers) or
    # when the volume column is missing. Exposes ONLY volume_zscore_20d;
    # gap_frequency_20d and intraday_range_percentile_252d (defined under
    # §1C at spec L299/L306 and re-surfaced in the §1E feature list at
    # spec L394-L397) live on volatility_state_v2.
    volume_liquidity_v2: VolumeLiquidityV2Features | None = None

    # V2 §2A monetary-pressure feature seam — populated when a
    # MonetaryPressureV2FeaturesConfig is threaded through AND
    # MarketContext.macro_series carries DGS2, DGS10, and broad_usd_index
    # (FRED DTWEXBGS). Includes broad_usd_index_zscore_63d and 21d-variant
    # features feeding the §2A axis classifier.
    monetary: MonetaryPressureV2Features | None = None

    # V2 §6.1 HMM evidence seam — populated when ``context.config.hmm``
    # is non-None AND the two V2 upstream seams gate locally:
    # volume_liquidity_v2 and network_fragility. The remaining three
    # inputs (volatility.return_1d, SPY-derived realized_vol_21d, and
    # drawdown_63d computed inline) ride the V1 path and are required
    # rather than optional. Otherwise None — V1 byte-identity preserved
    # on the 5-component transition_score path.
    hmm: HMMFeatures | None = None

    # v2 §6.2 GMM clustering evidence seam — populated when
    # ``context.config.clustering`` is non-None AND the seven §6.2 inputs
    # are all available. Predicate gates on ``breadth_state_v2.pct_above_50dma``
    # (PIT path lit), ``network_fragility``, and ``trend_direction_v2``;
    # ``trend_character`` + SPY-derived ``realized_vol_21d`` /
    # ``drawdown_63d`` ride the V1 path so they're always available. When
    # the seam is None, ``RegimeOutput.cluster`` is None (omitted on JSON
    # wire) and V1 byte-identity is preserved.
    clustering: ClusteringFeatures | None = None

    # v2 §6.3 BOCPD change-point evidence seam — populated when
    # ``context.config.change_point`` is non-None. The observation series
    # (SPY-derived realized_vol_21d) rides the V1 path; trailing-window
    # adequacy is enforced inside compute_change_point_features rather
    # than at this gate, so this seam only goes None when the config is
    # absent (v1-only callers). Consumed by the transition_score
    # 7-component weight table when present.
    change_point: ChangePointFeatures | None = None

    # V2 §2C credit/funding seam — populated when a CreditFundingConfig
    # is threaded through AND cross_asset_closes carries HYG/LQD/TLT/KRE AND
    # macro_series carries SOFR/IORB/NFCI/broad_usd_index. OAS keys are optional
    # at this gate; when absent the real-OAS label is unknown/data-unavailable
    # and the ETF proxy can still drive credit_funding_effective_state.
    credit_funding: CreditFundingFeatures | None = None

    # V2 §2B inflation/growth seam — populated when an
    # InflationGrowthConfig is threaded through AND cross_asset_closes carries
    # DBC/TLT/XLY/XLI/XLP/XLU AND macro_series carries cpi_all_items /
    # pmi_manufacturing / dgs10. Otherwise None — V1 byte-identity preserved
    # because RegimeOutput.inflation_growth_state defaults to None.
    inflation_growth: InflationGrowthFeatures | None = None


def _build_feature_store_state(
    context: MarketContext,
    *,
    network_fragility_config: NetworkFragilityConfig | None = None,
    trend_direction_v2_config: TrendDirectionV2Config | None = None,
    volatility_state_v2_config: VolatilityV2Config | None = None,
    breadth_state_v2_config: BreadthV2Config | None = None,
    volume_liquidity_v2_config: VolumeLiquidityV2Config | None = None,
    monetary_pressure_v2_config: MonetaryPressureV2FeaturesConfig | None = None,
    credit_funding_config: CreditFundingConfig | None = None,
    inflation_growth_config: InflationGrowthConfig | None = None,
    central_bank_text_config: CentralBankTextConfig | None = None,
    news_sentiment_config: NewsSentimentConfig | None = None,
    sentiment_score_config: SentimentScoreConfig | None = None,
) -> FeatureStoreBuildState:
    spy_ohlcv = context.spy_ohlcv
    spy_close = series_column(spy_ohlcv, "close")
    effective_sentiment_score_config = (
        sentiment_score_config
        if sentiment_score_config is not None
        else context.config.sentiment_score
    )
    return FeatureStoreBuildState(
        context=context,
        spy_ohlcv=spy_ohlcv,
        spy_close=spy_close,
        network_fragility_config=network_fragility_config,
        trend_direction_v2_config=trend_direction_v2_config,
        volatility_state_v2_config=volatility_state_v2_config,
        breadth_state_v2_config=breadth_state_v2_config,
        volume_liquidity_v2_config=volume_liquidity_v2_config,
        monetary_pressure_v2_config=monetary_pressure_v2_config,
        credit_funding_config=credit_funding_config,
        inflation_growth_config=inflation_growth_config,
        central_bank_text_config=central_bank_text_config,
        news_sentiment_config=news_sentiment_config,
        sentiment_score_config=effective_sentiment_score_config,
    )


def _feature_store_from_build_state(
    build_state: FeatureStoreBuildState,
    availability: dict[str, FeatureAvailability],
) -> FeatureStore:
    return FeatureStore(
        spy_index=as_datetime_index(build_state.spy_ohlcv.index),
        availability=availability,
        trend_direction=require_feature(build_state.trend_direction, "trend_direction"),
        trend_character=require_feature(build_state.trend_character, "trend_character"),
        volatility=require_feature(build_state.volatility, "volatility"),
        breadth=require_feature(build_state.breadth, "breadth"),
        sma_50=require_feature(build_state.sma_50, "sma_50"),
        network_fragility=build_state.network_fragility,
        trend_direction_v2=build_state.trend_direction_v2,
        volatility_state_v2=build_state.volatility_state_v2,
        breadth_state_v2=build_state.breadth_state_v2,
        volume_liquidity_v2=build_state.volume_liquidity_v2,
        monetary=build_state.monetary,
        hmm=build_state.hmm,
        clustering=build_state.clustering,
        change_point=build_state.change_point,
        credit_funding=build_state.credit_funding,
        inflation_growth=build_state.inflation_growth,
    )


def build_feature_store(
    context: MarketContext,
    *,
    network_fragility_config: NetworkFragilityConfig | None = None,
    trend_direction_v2_config: TrendDirectionV2Config | None = None,
    volatility_state_v2_config: VolatilityV2Config | None = None,
    breadth_state_v2_config: BreadthV2Config | None = None,
    volume_liquidity_v2_config: VolumeLiquidityV2Config | None = None,
    monetary_pressure_v2_config: MonetaryPressureV2FeaturesConfig | None = None,
    credit_funding_config: CreditFundingConfig | None = None,
    inflation_growth_config: InflationGrowthConfig | None = None,
    central_bank_text_config: CentralBankTextConfig | None = None,
    news_sentiment_config: NewsSentimentConfig | None = None,
    sentiment_score_config: SentimentScoreConfig | None = None,
) -> FeatureStore:
    build_state = _build_feature_store_state(
        context,
        network_fragility_config=network_fragility_config,
        trend_direction_v2_config=trend_direction_v2_config,
        volatility_state_v2_config=volatility_state_v2_config,
        breadth_state_v2_config=breadth_state_v2_config,
        volume_liquidity_v2_config=volume_liquidity_v2_config,
        monetary_pressure_v2_config=monetary_pressure_v2_config,
        credit_funding_config=credit_funding_config,
        inflation_growth_config=inflation_growth_config,
        central_bank_text_config=central_bank_text_config,
        news_sentiment_config=news_sentiment_config,
        sentiment_score_config=sentiment_score_config,
    )
    availability = _run_feature_specs(FEATURE_SPECS, build_state)
    return _feature_store_from_build_state(build_state, availability)
