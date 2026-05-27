from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection._rolling_stats import simple_moving_average
from regime_detection.breadth_state import (
    BreadthFeatures,
    compute_features as compute_breadth_features,
)
from regime_detection.breadth_state_v2 import (
    BreadthV2Features,
    compute_breadth_v2_features,
)
from regime_detection.change_point import (
    ChangePointFeatures,
    compute_change_point_features,
)
from regime_detection.central_bank_text import to_daily_score_series
from regime_detection.config import (
    BreadthV2Config,
    CentralBankTextConfig,
    CreditFundingConfig,
    InflationGrowthConfig,
    MonetaryPressureV2FeaturesConfig,
    NetworkFragilityConfig,
    NewsSentimentConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
    VolumeLiquidityV2Config,
)
from regime_detection.credit_funding import (
    CreditFundingFeatures,
    HYG_KEY as _CF_HYG_KEY,
    LQD_KEY as _CF_LQD_KEY,
    REQUIRED_CROSS_ASSET_KEYS as _CF_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as _CF_MACRO_KEYS,
    TLT_KEY as _CF_TLT_KEY,
    KRE_KEY as _CF_KRE_KEY,
    SOFR_KEY as _CF_SOFR_KEY,
    IORB_KEY as _CF_IORB_KEY,
    FEDFUNDS_KEY as _CF_FEDFUNDS_KEY,
    IOER_LEGACY_KEY as _CF_IOER_LEGACY_KEY,
    NFCI_KEY as _CF_NFCI_KEY,
    BROAD_USD_INDEX_KEY as _CF_BROAD_USD_KEY,
    HY_OAS_KEY as _CF_HY_OAS_KEY,
    IG_OAS_KEY as _CF_IG_OAS_KEY,
    compute_credit_funding_features,
)
from regime_detection.event_calendar import compute_event_window_just_passed
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
    compute_trailing_drawdown,
    compute_trend_v2_features,
)
from regime_detection.clustering import (
    ClusteringFeatures,
    compute_clustering_features,
)
from regime_detection.hmm_state import HMMFeatures, compute_hmm_features
from regime_detection.volatility_state import realized_vol
from regime_detection.volatility_state import (
    VolatilityFeatures,
    compute_features as compute_volatility_features,
)
from regime_detection.volatility_state_v2 import (
    VolatilityV2Features,
    compute_volatility_v2_features,
)
from regime_detection.volume_liquidity_v2 import (
    VolumeLiquidityV2Features,
    compute_volume_liquidity_v2_features,
)
from regime_detection.monetary_pressure import (
    MonetaryPressureV2Features,
    compute_monetary_pressure_features,
)
from regime_detection.inflation_growth import (
    AGG_FORWARD_EPS_REVISION_KEY as _IG_AGG_FORWARD_EPS_REVISION_KEY,
    CPI_KEY as _IG_CPI_KEY,
    CPI_NOWCAST_KEY as _IG_CPI_NOWCAST_KEY,
    DBC_KEY as _IG_DBC_KEY,
    DGS10_KEY as _IG_DGS10_KEY,
    InflationGrowthFeatures,
    PMI_KEY as _IG_PMI_KEY,
    REQUIRED_CROSS_ASSET_KEYS as _IG_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as _IG_MACRO_KEYS,
    TLT_KEY as _IG_TLT_KEY,
    XLI_KEY as _IG_XLI_KEY,
    XLP_KEY as _IG_XLP_KEY,
    XLU_KEY as _IG_XLU_KEY,
    XLY_KEY as _IG_XLY_KEY,
    compute_inflation_growth_features,
)
from regime_detection.feature_store_runtime import (
    FeatureAvailability,
    FeatureAvailabilityPolicy,
    FeatureSpec,
    _run_feature_specs,
)

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


_FRED_DGS2_KEY = "2y_yield"
_T = TypeVar("_T")


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise RuntimeError("feature store requires a DatetimeIndex-backed SPY frame")
    return index


def _series_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column]


@dataclass
class _FeatureStoreBuildState:
    context: MarketContext
    spy_ohlcv: pd.DataFrame
    spy_close: pd.Series
    network_fragility_config: NetworkFragilityConfig | None = None
    trend_direction_v2_config: TrendDirectionV2Config | None = None
    volatility_state_v2_config: VolatilityV2Config | None = None
    breadth_state_v2_config: BreadthV2Config | None = None
    volume_liquidity_v2_config: VolumeLiquidityV2Config | None = None
    monetary_pressure_v2_config: MonetaryPressureV2FeaturesConfig | None = None
    credit_funding_config: CreditFundingConfig | None = None
    inflation_growth_config: InflationGrowthConfig | None = None
    central_bank_text_config: CentralBankTextConfig | None = None
    news_sentiment_config: NewsSentimentConfig | None = None
    trend_direction: TrendDirectionFeatures | None = None
    sentiment_score: pd.Series | None = None
    news_sentiment_score: pd.Series | None = None
    trend_direction_v2: TrendDirectionV2Features | None = None
    trend_character: TrendCharacterFeatures | None = None
    volatility: VolatilityFeatures | None = None
    breadth: BreadthFeatures | None = None
    sma_50: pd.Series | None = None
    network_fragility: NetworkFragilityFeatures | None = None
    volatility_state_v2: VolatilityV2Features | None = None
    breadth_state_v2: BreadthV2Features | None = None
    volume_liquidity_v2: VolumeLiquidityV2Features | None = None
    monetary: MonetaryPressureV2Features | None = None
    realized_vol_21d: pd.Series | None = None
    # SPY-derived 63d trailing drawdown — spec §6.1 line 4059 input shared
    # by HMM (_build_hmm_feature) and clustering (_build_clustering_feature).
    # Memoized here so both consumers read the same series (cf. realized_vol_21d).
    drawdown_63d: pd.Series | None = None
    hmm: HMMFeatures | None = None
    clustering: ClusteringFeatures | None = None
    credit_funding: CreditFundingFeatures | None = None
    inflation_growth: InflationGrowthFeatures | None = None
    change_point: ChangePointFeatures | None = None


def _require_feature(value: _T | None, name: str) -> _T:
    if value is None:
        raise RuntimeError(f"feature builder did not populate required feature: {name}")
    return value


@dataclass(frozen=True)
class _FeatureStoreBuilder:
    name: str
    build: Callable[[_FeatureStoreBuildState], None]


def _run_feature_store_builders(
    builders: tuple[_FeatureStoreBuilder, ...],
    state: _FeatureStoreBuildState,
) -> None:
    for builder in builders:
        builder.build(state)


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


def _build_sentiment_score_series(
    *,
    aaii_sentiment: pd.DataFrame | None,
    session_index: pd.DatetimeIndex,
) -> pd.Series | None:
    """Align AAII bull-bear-spread 8w-MA onto the SPY session index for
    consumption by the v2 §1A `euphoria` predicate (ADR 0004 Q1+Q4).

    Forward-fill semantics: use the latest AAII row whose
    ``publication_date`` (or ``date`` if no separate publication column) is
    on or before each session — V1 §2.2 stateless-replay rule, never
    consult a future-dated reading. Returns ``None`` when no AAII frame
    is supplied (lets the euphoria predicate falsify per the V2 §10
    absolute "do not invent" rule at spec L4364 and the documented
    implementation decision lineage in ADR 0004 Q1+Q4).

    Cold-start (ADR 0004 Q5): a session with no AAII row at or before it
    receives NaN; the euphoria predicate then falsifies on that session
    per V1 §2.7. With the AAII fetcher's `min_periods=1` 8-week MA, the
    output column is populated from week 1 of the data, so the practical
    cold-start window is just "no AAII history yet."
    """
    if aaii_sentiment is None:
        return None
    if aaii_sentiment.empty:
        return None
    if "bull_bear_spread_8w_ma" not in aaii_sentiment.columns:
        return None
    publication_column = (
        "publication_date" if "publication_date" in aaii_sentiment.columns else "date"
    )
    if publication_column not in aaii_sentiment.columns:
        return None
    sorted_aaii = aaii_sentiment.sort_values(publication_column).reset_index(drop=True)
    publication = pd.to_datetime(sorted_aaii[publication_column])
    score_values = sorted_aaii["bull_bear_spread_8w_ma"].astype(float).to_numpy()
    aligned = pd.Series(
        score_values,
        index=pd.DatetimeIndex(publication),
        name="sentiment_score",
    )
    # Reindex forward-fill onto the SPY session calendar (each session gets
    # the value of the latest AAII publication on or before it).
    return aligned.reindex(session_index, method="ffill")


def _build_news_sentiment_score_series(
    *,
    news_sentiment: pd.Series | None,
    session_index: pd.DatetimeIndex,
    config: NewsSentimentConfig | None,
) -> pd.Series | None:
    if config is None:
        return None
    if news_sentiment is None:
        return None
    if news_sentiment.empty:
        return None
    score = (
        news_sentiment.reindex(session_index, method="ffill")
        .rolling(config.smoothing_window_sessions, min_periods=1)
        .mean()
    )
    score.name = "news_sentiment_score"
    return score


def _build_sentiment_score_feature(state: _FeatureStoreBuildState) -> None:
    state.sentiment_score = _build_sentiment_score_series(
        aaii_sentiment=state.context.aaii_sentiment,
        session_index=_as_datetime_index(state.spy_close.index),
    )


def _build_news_sentiment_score_feature(state: _FeatureStoreBuildState) -> None:
    state.news_sentiment_score = _build_news_sentiment_score_series(
        news_sentiment=state.context.news_sentiment,
        session_index=_as_datetime_index(state.spy_close.index),
        config=state.news_sentiment_config,
    )


def _build_trend_direction_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.trend_direction_v2_config is None:
        state.trend_direction_v2 = None
        return
    state.trend_direction_v2 = compute_trend_v2_features(
        state.spy_close,
        config=state.trend_direction_v2_config,
        sentiment_score=state.sentiment_score,
        news_sentiment_score=state.news_sentiment_score,
    )


def _build_network_fragility_feature(state: _FeatureStoreBuildState) -> None:
    if state.context.sector_etf_closes is None:
        state.network_fragility = None
        return
    if state.network_fragility_config is None:
        state.network_fragility = compute_network_fragility_features(
            sector_etf_closes=state.context.sector_etf_closes,
            cross_asset_closes=state.context.cross_asset_closes or {},
            spy_close=state.spy_close,
        )
        return
    config = state.network_fragility_config
    state.network_fragility = compute_network_fragility_features(
        sector_etf_closes=state.context.sector_etf_closes,
        cross_asset_closes=state.context.cross_asset_closes or {},
        spy_close=state.spy_close,
        correlation_lookback_days=config.correlation_lookback_days,
        percentile_lookback_days=config.percentile_lookback_days,
        realized_vol_lookback_days=config.realized_vol_lookback_days,
        dispersion_percentile_lookback_days=config.dispersion_percentile_lookback_days,
        dispersion_spy_vol_floor=config.dispersion_spy_vol_floor,
        min_universe_size=config.min_universe_size,
        min_window_completeness=config.min_window_completeness,
        universe=config.universe,
    )


def _build_volatility_state_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.volatility_state_v2_config is None:
        state.volatility_state_v2 = None
        return
    event_window = (
        compute_event_window_just_passed(
            normalized_event_calendar=state.context.normalized_event_calendar,
            sessions=tuple(
                ts.date() for ts in _as_datetime_index(state.spy_close.index)
            ),
            trailing_sessions=(
                state.volatility_state_v2_config.rules.vol_crush_event_window_trailing_sessions
            ),
        )
        if state.context.normalized_event_calendar is not None
        else None
    )
    state.volatility_state_v2 = compute_volatility_v2_features(
        open_=_series_column(state.spy_ohlcv, "open"),
        high=_series_column(state.spy_ohlcv, "high"),
        low=_series_column(state.spy_ohlcv, "low"),
        close=state.spy_close,
        config=state.volatility_state_v2_config,
        rules_config=state.volatility_state_v2_config.rules,
        implied_vol_30d=state.context.implied_vol_30d,
        event_window_just_passed=event_window,
    )


def _build_breadth_state_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.breadth_state_v2_config is None or state.context.sector_etf_closes is None:
        state.breadth_state_v2 = None
        return
    sector_closes = state.context.sector_etf_closes
    if not any(symbol in sector_closes for symbol in SECTOR_ETFS):
        state.breadth_state_v2 = None
        return
    state.breadth_state_v2 = compute_breadth_v2_features(
        sector_etf_closes=sector_closes,
        config=state.breadth_state_v2_config,
        pit_constituent_intervals=state.context.pit_constituent_intervals,
        constituent_ohlcv=state.context.constituent_ohlcv,
    )


def _build_volume_liquidity_v2_feature(state: _FeatureStoreBuildState) -> None:
    spy_volume = (
        _series_column(state.spy_ohlcv, "volume")
        if "volume" in state.spy_ohlcv.columns
        else None
    )
    if (
        state.volume_liquidity_v2_config is None
        or spy_volume is None
        or bool(spy_volume.isna().all())
    ):
        state.volume_liquidity_v2 = None
        return
    state.volume_liquidity_v2 = compute_volume_liquidity_v2_features(
        volume=spy_volume,
        config=state.volume_liquidity_v2_config,
    )


def _build_monetary_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.monetary_pressure_v2_config is None
        or state.context.macro_series is None
        or _FRED_DGS2_KEY not in state.context.macro_series
        or _IG_DGS10_KEY not in state.context.macro_series
        or "broad_usd_index" not in state.context.macro_series
    ):
        state.monetary = None
        return
    broad_usd_series = state.context.macro_series["broad_usd_index"]
    cb_text_score_series: pd.Series | None = None
    if (
        state.central_bank_text_config is not None
        and state.context.central_bank_text_releases is not None
        and not state.context.central_bank_text_releases.empty
    ):
        cb_text_score_series = to_daily_score_series(
            state.context.central_bank_text_releases,
            session_index=_as_datetime_index(state.spy_close.index),
            smoothing_window_sessions=state.central_bank_text_config.smoothing_window_sessions,
            same_date_aggregation=state.central_bank_text_config.same_date_aggregation,
            max_release_age_days=state.central_bank_text_config.max_release_age_days,
        )
    state.monetary = compute_monetary_pressure_features(
        dgs2=state.context.macro_series[_FRED_DGS2_KEY],
        dgs10=state.context.macro_series[_IG_DGS10_KEY],
        broad_usd_index=broad_usd_series,
        central_bank_text_score=cb_text_score_series,
        config=state.monetary_pressure_v2_config,
    )


def _build_realized_vol_21d_feature(state: _FeatureStoreBuildState) -> None:
    state.realized_vol_21d = (
        realized_vol(state.spy_close, 21)
        if (
            state.context.config.hmm is not None
            or state.context.config.clustering is not None
            or state.context.config.change_point is not None
        )
        else None
    )


def _build_drawdown_63d_feature(state: _FeatureStoreBuildState) -> None:
    # Spec §6.1 line 4059: 63d trailing-peak drawdown. Single shared
    # computation matches the realized_vol_21d pattern (one helper, two
    # consumers — HMM and clustering).
    state.drawdown_63d = (
        compute_trailing_drawdown(state.spy_close, 63)
        if (
            state.context.config.hmm is not None
            or state.context.config.clustering is not None
        )
        else None
    )


def _build_hmm_feature(state: _FeatureStoreBuildState) -> None:
    volume_liquidity_v2 = state.volume_liquidity_v2
    network_fragility = state.network_fragility
    if (
        state.context.config.hmm is None
        or volume_liquidity_v2 is None
        or network_fragility is None
    ):
        state.hmm = None
        return
    volatility = _require_feature(state.volatility, "volatility")
    state.hmm = compute_hmm_features(
        return_1d=volatility.return_1d,
        realized_vol_21d=state.realized_vol_21d,
        drawdown_63d=state.drawdown_63d,
        volume_zscore_20d=volume_liquidity_v2.volume_zscore_20d,
        avg_pairwise_corr_63d=network_fragility.avg_pairwise_corr_63d,
        config=state.context.config.hmm,
    )


def _build_clustering_feature(state: _FeatureStoreBuildState) -> None:
    breadth_state_v2 = state.breadth_state_v2
    network_fragility = state.network_fragility
    trend_direction_v2 = state.trend_direction_v2
    if (
        state.context.config.clustering is None
        or breadth_state_v2 is None
        or breadth_state_v2.pct_above_50dma is None
        or network_fragility is None
        or trend_direction_v2 is None
    ):
        state.clustering = None
        return
    trend_character = _require_feature(state.trend_character, "trend_character")
    state.clustering = compute_clustering_features(
        return_21d=trend_character.return_21d,
        return_63d=trend_direction_v2.return_63d,
        realized_vol_21d=state.realized_vol_21d,
        drawdown_63d=state.drawdown_63d,
        adx_14=trend_character.adx_14,
        avg_pairwise_corr_63d=network_fragility.avg_pairwise_corr_63d,
        pct_above_50dma=breadth_state_v2.pct_above_50dma,
        config=state.context.config.clustering,
    )


def _build_credit_funding_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.credit_funding_config is None
        or state.context.cross_asset_closes is None
        or state.context.macro_series is None
        or not all(k in state.context.cross_asset_closes for k in _CF_CROSS_ASSET_KEYS)
        or not all(k in state.context.macro_series for k in _CF_MACRO_KEYS)
    ):
        state.credit_funding = None
        return
    nan_oas = pd.Series(float("nan"), index=state.spy_close.index)
    state.credit_funding = compute_credit_funding_features(
        hyg_close=state.context.cross_asset_closes[_CF_HYG_KEY],
        lqd_close=state.context.cross_asset_closes[_CF_LQD_KEY],
        tlt_close=state.context.cross_asset_closes[_CF_TLT_KEY],
        kre_close=state.context.cross_asset_closes[_CF_KRE_KEY],
        spy_close=state.spy_close,
        sofr=state.context.macro_series[_CF_SOFR_KEY],
        iorb=state.context.macro_series[_CF_IORB_KEY],
        nfci_weekly=state.context.macro_series[_CF_NFCI_KEY],
        broad_usd_index=state.context.macro_series[_CF_BROAD_USD_KEY],
        hy_oas=state.context.macro_series.get(_CF_HY_OAS_KEY, nan_oas),
        ig_oas=state.context.macro_series.get(_CF_IG_OAS_KEY, nan_oas),
        config=state.credit_funding_config.rules,
        fedfunds=state.context.macro_series.get(_CF_FEDFUNDS_KEY),
        ioer_legacy=state.context.macro_series.get(_CF_IOER_LEGACY_KEY),
    )


def _build_inflation_growth_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.inflation_growth_config is None
        or state.context.cross_asset_closes is None
        or state.context.macro_series is None
        or not all(k in state.context.cross_asset_closes for k in _IG_CROSS_ASSET_KEYS)
        or not all(k in state.context.macro_series for k in _IG_MACRO_KEYS)
    ):
        state.inflation_growth = None
        return
    state.inflation_growth = compute_inflation_growth_features(
        cpi_all_items=state.context.macro_series[_IG_CPI_KEY],
        pmi_manufacturing=state.context.macro_series[_IG_PMI_KEY],
        dgs10=state.context.macro_series[_IG_DGS10_KEY],
        dbc_close=state.context.cross_asset_closes[_IG_DBC_KEY],
        spy_close=state.spy_close,
        tlt_close=state.context.cross_asset_closes[_IG_TLT_KEY],
        xly_close=state.context.cross_asset_closes[_IG_XLY_KEY],
        xli_close=state.context.cross_asset_closes[_IG_XLI_KEY],
        xlp_close=state.context.cross_asset_closes[_IG_XLP_KEY],
        xlu_close=state.context.cross_asset_closes[_IG_XLU_KEY],
        config=state.inflation_growth_config.rules,
        cpi_nowcast=state.context.macro_series.get(_IG_CPI_NOWCAST_KEY),
        aggregate_forward_eps_revision=state.context.macro_series.get(
            _IG_AGG_FORWARD_EPS_REVISION_KEY
        ),
        cpi_first_release=state.context.cpi_first_release,
        use_first_release_cpi_when_available=(
            state.inflation_growth_config.rules.use_first_release_cpi_when_available
        ),
    )


def _build_change_point_feature(state: _FeatureStoreBuildState) -> None:
    if state.context.config.change_point is None:
        state.change_point = None
        return
    state.change_point = compute_change_point_features(
        realized_vol_21d=state.realized_vol_21d,
        config=state.context.config.change_point,
    )


# --- New spec-based builders (PR 1) ------------------------------------------


def _build_trend_direction(spy_close: pd.Series) -> TrendDirectionFeatures:
    return compute_trend_direction_features(spy_close)


def _resolve_trend_direction(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {"spy_close": state.spy_close}


def _build_trend_character(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    tc_v2_config: object,
) -> TrendCharacterFeatures:
    if tc_v2_config is not None:
        return compute_trend_character_features(
            close=close,
            high=high,
            low=low,
            volume=volume,
            bb_width_period=tc_v2_config.bb_width_period,  # type: ignore[union-attr]
            bb_width_multiplier=tc_v2_config.bb_width_multiplier,  # type: ignore[union-attr]
            bb_width_expanding_lookback=tc_v2_config.bb_width_expanding_lookback,  # type: ignore[union-attr]
            followthrough_lookback_sessions=tc_v2_config.followthrough_lookback_sessions,  # type: ignore[union-attr]
            followthrough_window_count=tc_v2_config.followthrough_window_count,  # type: ignore[union-attr]
            followthrough_hold_sessions=tc_v2_config.followthrough_hold_sessions,  # type: ignore[union-attr]
        )
    return compute_trend_character_features(
        close=close, high=high, low=low, volume=volume
    )


def _resolve_trend_character(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    volume = (
        _series_column(state.spy_ohlcv, "volume")
        if "volume" in state.spy_ohlcv.columns
        else None
    )
    return {
        "close": state.spy_close,
        "high": _series_column(state.spy_ohlcv, "high"),
        "low": _series_column(state.spy_ohlcv, "low"),
        "volume": volume,
        "tc_v2_config": state.context.config.trend_character_v2,
    }


def _build_volatility(
    close: pd.Series, vix_proxy_close: pd.Series | None
) -> VolatilityFeatures:
    return compute_volatility_features(close=close, vix_proxy_close=vix_proxy_close)


def _resolve_volatility(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {
        "close": state.spy_close,
        "vix_proxy_close": state.context.vix_proxy_close,
    }


def _build_breadth(spy_close: pd.Series, rsp_close: pd.Series) -> BreadthFeatures:
    return compute_breadth_features(spy_close=spy_close, rsp_close=rsp_close)


def _resolve_breadth(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {
        "spy_close": state.spy_close,
        "rsp_close": state.context.rsp_close.reindex(state.spy_ohlcv.index),
    }


def _build_sma_50(spy_close: pd.Series) -> pd.Series:
    return simple_moving_average(spy_close, window=50)


def _resolve_sma_50(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {"spy_close": state.spy_close}


_FEATURE_SPECS: tuple[FeatureSpec[object, _FeatureStoreBuildState], ...] = (
    FeatureSpec(
        name="trend_direction",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_trend_direction,
        build=_build_trend_direction,
        store=lambda s, v: setattr(s, "trend_direction", v),
    ),
    FeatureSpec(
        name="trend_character",
        policy="raise",
        required_inputs=("spy_ohlcv.close", "spy_ohlcv.high", "spy_ohlcv.low"),
        resolve=_resolve_trend_character,
        build=_build_trend_character,
        store=lambda s, v: setattr(s, "trend_character", v),
    ),
    FeatureSpec(
        name="volatility",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_volatility,
        build=_build_volatility,
        store=lambda s, v: setattr(s, "volatility", v),
    ),
    FeatureSpec(
        name="breadth",
        policy="raise",
        required_inputs=("spy_ohlcv.close", "rsp_close"),
        resolve=_resolve_breadth,
        build=_build_breadth,
        store=lambda s, v: setattr(s, "breadth", v),
    ),
    FeatureSpec(
        name="sma_50",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_sma_50,
        build=_build_sma_50,
        store=lambda s, v: setattr(s, "sma_50", v),
    ),
)


_FEATURE_STORE_BUILDERS: tuple[_FeatureStoreBuilder, ...] = (
    _FeatureStoreBuilder("sentiment_score", _build_sentiment_score_feature),
    _FeatureStoreBuilder("news_sentiment_score", _build_news_sentiment_score_feature),
    _FeatureStoreBuilder("trend_direction_v2", _build_trend_direction_v2_feature),
    _FeatureStoreBuilder("network_fragility", _build_network_fragility_feature),
    _FeatureStoreBuilder("volatility_state_v2", _build_volatility_state_v2_feature),
    _FeatureStoreBuilder("breadth_state_v2", _build_breadth_state_v2_feature),
    _FeatureStoreBuilder("volume_liquidity_v2", _build_volume_liquidity_v2_feature),
    _FeatureStoreBuilder("monetary", _build_monetary_feature),
    _FeatureStoreBuilder("realized_vol_21d", _build_realized_vol_21d_feature),
    _FeatureStoreBuilder("drawdown_63d", _build_drawdown_63d_feature),
    _FeatureStoreBuilder("hmm", _build_hmm_feature),
    _FeatureStoreBuilder("clustering", _build_clustering_feature),
    _FeatureStoreBuilder("credit_funding", _build_credit_funding_feature),
    _FeatureStoreBuilder("inflation_growth", _build_inflation_growth_feature),
    _FeatureStoreBuilder("change_point", _build_change_point_feature),
)


def _missing_macro_keys(
    macro_series: dict[str, pd.Series] | None, required_keys: tuple[str, ...]
) -> tuple[str, ...]:
    if macro_series is None:
        return ("macro_series",) + required_keys
    return tuple(key for key in required_keys if key not in macro_series)


def _missing_cross_asset_keys(
    cross_asset_closes: dict[str, pd.Series] | None, required_keys: tuple[str, ...]
) -> tuple[str, ...]:
    if cross_asset_closes is None:
        return ("cross_asset_closes",) + required_keys
    return tuple(key for key in required_keys if key not in cross_asset_closes)


def _missing_sector_inputs(state: _FeatureStoreBuildState) -> tuple[str, ...]:
    sector_closes = state.context.sector_etf_closes
    if sector_closes is None:
        return ("sector_etf_closes",)
    if not any(symbol in sector_closes for symbol in SECTOR_ETFS):
        return ("sector_etf_closes.any_sector_etf",)
    return ()


def _availability(
    *,
    feature: str,
    value: object | None,
    policy: FeatureAvailabilityPolicy,
    required_inputs: tuple[str, ...],
    missing_inputs: tuple[str, ...],
) -> FeatureAvailability:
    if value is not None:
        return FeatureAvailability(
            feature=feature,
            available=True,
            policy=policy,
            reason="populated",
            required_inputs=required_inputs,
        )
    reason = "not_configured" if not missing_inputs else "missing_required_inputs"
    return FeatureAvailability(
        feature=feature,
        available=False,
        policy=policy,
        reason=reason,
        required_inputs=required_inputs,
        missing_inputs=missing_inputs,
    )


def _build_feature_availability_report(
    state: _FeatureStoreBuildState,
) -> dict[str, FeatureAvailability]:
    spy_volume_missing = ()
    if "volume" not in state.spy_ohlcv.columns:
        spy_volume_missing = ("spy_ohlcv.volume",)
    else:
        spy_volume = _series_column(state.spy_ohlcv, "volume")
        if bool(spy_volume.isna().all()):
            spy_volume_missing = ("spy_ohlcv.volume.non_nan",)

    breadth_state_missing = (
        ()
        if state.breadth_state_v2_config is not None
        else ("breadth_state_v2_config",)
    ) + _missing_sector_inputs(state)
    volume_liquidity_missing = (
        ()
        if state.volume_liquidity_v2_config is not None
        else ("volume_liquidity_v2_config",)
    ) + spy_volume_missing
    monetary_config_missing = (
        ()
        if state.monetary_pressure_v2_config is not None
        else ("monetary_pressure_v2_config",)
    )
    credit_config_missing = (
        () if state.credit_funding_config is not None else ("credit_funding_config",)
    )
    inflation_config_missing = (
        ()
        if state.inflation_growth_config is not None
        else ("inflation_growth_config",)
    )

    monetary_required = (
        "macro_series",
        _FRED_DGS2_KEY,
        _IG_DGS10_KEY,
        "broad_usd_index",
    )
    monetary_missing = ()
    if state.monetary_pressure_v2_config is None:
        monetary_missing = ()
    else:
        monetary_missing = _missing_macro_keys(
            state.context.macro_series,
            (_FRED_DGS2_KEY, _IG_DGS10_KEY, "broad_usd_index"),
        )

    credit_missing = ()
    if state.credit_funding_config is not None:
        credit_missing = _missing_cross_asset_keys(
            state.context.cross_asset_closes, tuple(_CF_CROSS_ASSET_KEYS)
        ) + _missing_macro_keys(state.context.macro_series, tuple(_CF_MACRO_KEYS))

    inflation_missing = ()
    if state.inflation_growth_config is not None:
        inflation_missing = _missing_cross_asset_keys(
            state.context.cross_asset_closes, tuple(_IG_CROSS_ASSET_KEYS)
        ) + _missing_macro_keys(state.context.macro_series, tuple(_IG_MACRO_KEYS))

    report = {
        "network_fragility": _availability(
            feature="network_fragility",
            value=state.network_fragility,
            policy="none",
            required_inputs=("sector_etf_closes",),
            missing_inputs=(
                ()
                if state.context.sector_etf_closes is not None
                else ("sector_etf_closes",)
            ),
        ),
        "trend_direction_v2": _availability(
            feature="trend_direction_v2",
            value=state.trend_direction_v2,
            policy="none",
            required_inputs=("trend_direction_v2_config", "spy_ohlcv.close"),
            missing_inputs=(
                ()
                if state.trend_direction_v2_config is not None
                else ("trend_direction_v2_config",)
            ),
        ),
        "volatility_state_v2": _availability(
            feature="volatility_state_v2",
            value=state.volatility_state_v2,
            policy="none",
            required_inputs=("volatility_state_v2_config", "spy_ohlcv.ohlc"),
            missing_inputs=(
                ()
                if state.volatility_state_v2_config is not None
                else ("volatility_state_v2_config",)
            ),
        ),
        "breadth_state_v2": _availability(
            feature="breadth_state_v2",
            value=state.breadth_state_v2,
            policy="none",
            required_inputs=("breadth_state_v2_config", "sector_etf_closes"),
            missing_inputs=breadth_state_missing,
        ),
        "volume_liquidity_v2": _availability(
            feature="volume_liquidity_v2",
            value=state.volume_liquidity_v2,
            policy="none",
            required_inputs=("volume_liquidity_v2_config", "spy_ohlcv.volume"),
            missing_inputs=volume_liquidity_missing,
        ),
        "monetary": _availability(
            feature="monetary",
            value=state.monetary,
            policy="raise",
            required_inputs=monetary_required,
            missing_inputs=monetary_config_missing + monetary_missing,
        ),
        "hmm": _availability(
            feature="hmm",
            value=state.hmm,
            policy="none",
            required_inputs=("hmm_config", "volume_liquidity_v2", "network_fragility"),
            missing_inputs=tuple(
                item
                for item, available in (
                    ("hmm_config", state.context.config.hmm is not None),
                    ("volume_liquidity_v2", state.volume_liquidity_v2 is not None),
                    ("network_fragility", state.network_fragility is not None),
                )
                if not available
            ),
        ),
        "clustering": _availability(
            feature="clustering",
            value=state.clustering,
            policy="none",
            required_inputs=(
                "clustering_config",
                "breadth_state_v2.pct_above_50dma",
                "network_fragility",
                "trend_direction_v2",
            ),
            missing_inputs=tuple(
                item
                for item, available in (
                    ("clustering_config", state.context.config.clustering is not None),
                    (
                        "breadth_state_v2.pct_above_50dma",
                        state.breadth_state_v2 is not None
                        and state.breadth_state_v2.pct_above_50dma is not None,
                    ),
                    ("network_fragility", state.network_fragility is not None),
                    ("trend_direction_v2", state.trend_direction_v2 is not None),
                )
                if not available
            ),
        ),
        "change_point": _availability(
            feature="change_point",
            value=state.change_point,
            policy="none",
            required_inputs=("change_point_config", "realized_vol_21d"),
            missing_inputs=(
                ()
                if state.context.config.change_point is not None
                else ("change_point_config",)
            ),
        ),
        "credit_funding": _availability(
            feature="credit_funding",
            value=state.credit_funding,
            policy="none",
            required_inputs=(
                "credit_funding_config",
                "cross_asset_closes",
                "macro_series",
            ),
            missing_inputs=credit_config_missing + credit_missing,
        ),
        "inflation_growth": _availability(
            feature="inflation_growth",
            value=state.inflation_growth,
            policy="none",
            required_inputs=(
                "inflation_growth_config",
                "cross_asset_closes",
                "macro_series",
            ),
            missing_inputs=inflation_config_missing + inflation_missing,
        ),
    }
    return report


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
) -> FeatureStore:
    # TODO(refactor, owner=regime-maintainers): Decompose this builder in a dedicated no-behavior-change
    # refactor. Keep feature wiring and fixture replay frozen while extracting
    # helpers so classifier changes do not hide inside the decomposition.
    spy_ohlcv = context.spy_ohlcv
    spy_close = _series_column(spy_ohlcv, "close")
    build_state = _FeatureStoreBuildState(
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
    )
    spec_availability = _run_feature_specs(_FEATURE_SPECS, build_state)
    _run_feature_store_builders(_FEATURE_STORE_BUILDERS, build_state)
    legacy_availability = _build_feature_availability_report(build_state)
    combined_availability = {**spec_availability, **legacy_availability}

    return FeatureStore(
        spy_index=_as_datetime_index(spy_ohlcv.index),
        availability=combined_availability,
        trend_direction=_require_feature(
            build_state.trend_direction, "trend_direction"
        ),
        trend_character=_require_feature(
            build_state.trend_character, "trend_character"
        ),
        volatility=_require_feature(build_state.volatility, "volatility"),
        breadth=_require_feature(build_state.breadth, "breadth"),
        sma_50=_require_feature(build_state.sma_50, "sma_50"),
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
