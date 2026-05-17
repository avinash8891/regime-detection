from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection.breadth_state import BreadthFeatures, compute_features as compute_breadth_features
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
from regime_detection.volatility_state import VolatilityFeatures, compute_features as compute_volatility_features
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
    CPI_KEY as _IG_CPI_KEY,
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

__all__ = [
    "BreadthV2Features",
    "ChangePointFeatures",
    "ClusteringFeatures",
    "CreditFundingFeatures",
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


# v2 §2A source contract (lines 887–889). Pinned here so the feature
# store can detect whether MarketContext.macro_series carries the two
# required FRED series without scattering string literals.
_FRED_DGS2_KEY = "DGS2"
_FRED_DGS10_KEY = "DGS10"


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
    values: dict[str, Any] = field(default_factory=dict)


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

    # V2 §1E seam — populated when a VolumeLiquidityV2Config is threaded
    # through AND a SPY volume series is available on the context. SPY
    # volume rides on MarketContext.spy_ohlcv["volume"] on the V1+V2 path
    # so this is only None when the config is absent (v1-only callers) or
    # when the volume column is missing. Exposes ONLY volume_zscore_20d;
    # gap_frequency_20d and intraday_range_percentile_252d (also §1E
    # features per spec lines 257–258) live on volatility_state_v2.
    volume_liquidity_v2: VolumeLiquidityV2Features | None = None

    # V2 §2A seam (Slice 4.1, evidence-only) — populated when a
    # MonetaryPressureV2FeaturesConfig is threaded through AND
    # MarketContext.macro_series carries both DGS2 and DGS10 (spec
    # source contract §2A lines 887–889). Exposes ONLY the two
    # spec-pinned yield z-scores; broad_usd_index and 21d-variant
    # features are deferred per Ambiguity Log #44 / #45.
    monetary: MonetaryPressureV2Features | None = None

    # V2 §6.1 HMM evidence seam (Slice 6) — populated when ``context.config.hmm``
    # is non-None AND all five upstream feature seams are lit (volatility
    # return_1d, volume_liquidity_v2.volume_zscore_20d,
    # network_fragility.avg_pairwise_corr_63d, plus the SPY-derived
    # realized_vol_21d and drawdown_63d). Otherwise None — V1 byte-identity
    # preserved on the 5-component transition_score path.
    hmm: HMMFeatures | None = None

    # v2 §6.2 GMM clustering evidence seam (Slice 7) — populated when
    # ``context.config.clustering`` is non-None AND the seven §6.2 inputs
    # are all available. Predicate gates on ``breadth_state_v2.pct_above_50dma``
    # (PIT path lit), ``network_fragility``, and ``trend_direction_v2``;
    # ``trend_character`` + SPY-derived ``realized_vol_21d`` /
    # ``drawdown_63d`` ride the V1 path so they're always available. When
    # the seam is None, ``RegimeOutput.cluster`` is None (omitted on JSON
    # wire) and V1 byte-identity is preserved.
    clustering: ClusteringFeatures | None = None

    # v2 §6.3 BOCPD change-point evidence seam (Slice 8) — populated when
    # ``context.config.change_point`` is non-None AND the trailing
    # ``training_window_days`` of realized_vol_21d (SPY-derived) is
    # available. The observation series rides the SPY-close V1 path, so
    # this seam only goes None when the config is absent (v1-only callers)
    # or when SPY history is too short for the spec-pinned 5y window. The
    # transition_score consumer is V2.1 spec-amendment work — Slice 8 is
    # evidence-only and does NOT change the V1 5-component score path.
    change_point: ChangePointFeatures | None = None

    # V2 §2C credit/funding seam (Slice 4) — populated when a CreditFundingConfig
    # is threaded through AND cross_asset_closes carries HYG/LQD/TLT/KRE AND
    # macro_series carries SOFR/IORB/NFCI/broad_usd_index. OAS keys are optional
    # at this gate; when absent the real-OAS label is unknown/data-unavailable
    # and the ETF proxy can still drive credit_funding_effective_state.
    credit_funding: CreditFundingFeatures | None = None

    # V2 §2B inflation/growth seam (Slice 5) — populated when an
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
    is supplied (lets the euphoria predicate falsify per V2 §10 "do not
    invent a sentiment proxy" / Log #32 lineage).

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
    return (
        news_sentiment.reindex(session_index, method="ffill")
        .rolling(config.smoothing_window_sessions, min_periods=1)
        .mean()
        .rename("news_sentiment_score")
    )


def _build_trend_direction_feature(state: _FeatureStoreBuildState) -> None:
    state.values["trend_direction"] = compute_trend_direction_features(state.spy_close)


def _build_sentiment_score_feature(state: _FeatureStoreBuildState) -> None:
    state.values["sentiment_score"] = _build_sentiment_score_series(
        aaii_sentiment=state.context.aaii_sentiment,
        session_index=state.spy_close.index,
    )


def _build_news_sentiment_score_feature(state: _FeatureStoreBuildState) -> None:
    state.values["news_sentiment_score"] = _build_news_sentiment_score_series(
        news_sentiment=state.context.news_sentiment,
        session_index=state.spy_close.index,
        config=state.news_sentiment_config,
    )


def _build_trend_direction_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.trend_direction_v2_config is None:
        state.values["trend_direction_v2"] = None
        return
    state.values["trend_direction_v2"] = compute_trend_v2_features(
        state.spy_close,
        config=state.trend_direction_v2_config,
        sentiment_score=state.values.get("sentiment_score"),
        news_sentiment_score=state.values.get("news_sentiment_score"),
    )


def _build_trend_character_feature(state: _FeatureStoreBuildState) -> None:
    state.values["trend_character"] = compute_trend_character_features(
        close=state.spy_close,
        high=state.spy_ohlcv["high"],
        low=state.spy_ohlcv["low"],
        volume=state.spy_ohlcv["volume"] if "volume" in state.spy_ohlcv.columns else None,
    )


def _build_volatility_feature(state: _FeatureStoreBuildState) -> None:
    state.values["volatility"] = compute_volatility_features(
        close=state.spy_close,
        vix_proxy_close=state.context.vix_proxy_close,
    )


def _build_breadth_feature(state: _FeatureStoreBuildState) -> None:
    state.values["breadth"] = compute_breadth_features(
        spy_close=state.spy_close,
        rsp_close=state.context.rsp_close.reindex(state.spy_ohlcv.index),
    )


def _build_sma_50_feature(state: _FeatureStoreBuildState) -> None:
    state.values["sma_50"] = state.spy_close.rolling(50).mean()


def _build_network_fragility_feature(state: _FeatureStoreBuildState) -> None:
    if state.context.sector_etf_closes is None:
        state.values["network_fragility"] = None
        return
    nf_kwargs: dict[str, int | float] = {}
    if state.network_fragility_config is not None:
        nf_kwargs = {
            "correlation_lookback_days": state.network_fragility_config.correlation_lookback_days,
            "percentile_lookback_days": state.network_fragility_config.percentile_lookback_days,
            "realized_vol_lookback_days": state.network_fragility_config.realized_vol_lookback_days,
            "dispersion_percentile_lookback_days": (
                state.network_fragility_config.dispersion_percentile_lookback_days
            ),
            "min_universe_size": state.network_fragility_config.min_universe_size,
            "min_window_completeness": state.network_fragility_config.min_window_completeness,
        }
    state.values["network_fragility"] = compute_network_fragility_features(
        sector_etf_closes=state.context.sector_etf_closes,
        cross_asset_closes=state.context.cross_asset_closes or {},
        spy_close=state.spy_close,
        **nf_kwargs,
    )


def _build_volatility_state_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.volatility_state_v2_config is None:
        state.values["volatility_state_v2"] = None
        return
    event_window = (
        compute_event_window_just_passed(
            normalized_event_calendar=state.context.normalized_event_calendar,
            sessions=tuple(state.spy_close.index.date),
            trailing_sessions=(
                state.volatility_state_v2_config.rules.vol_crush_event_window_trailing_sessions
            ),
        )
        if state.context.normalized_event_calendar is not None
        else None
    )
    state.values["volatility_state_v2"] = compute_volatility_v2_features(
        open_=state.spy_ohlcv["open"],
        high=state.spy_ohlcv["high"],
        low=state.spy_ohlcv["low"],
        close=state.spy_close,
        config=state.volatility_state_v2_config,
        rules_config=state.volatility_state_v2_config.rules,
        implied_vol_30d=state.context.implied_vol_30d,
        event_window_just_passed=event_window,
    )


def _build_breadth_state_v2_feature(state: _FeatureStoreBuildState) -> None:
    if state.breadth_state_v2_config is None or state.context.sector_etf_closes is None:
        state.values["breadth_state_v2"] = None
        return
    sector_closes = state.context.sector_etf_closes
    if not all(symbol in sector_closes for symbol in SECTOR_ETFS):
        state.values["breadth_state_v2"] = None
        return
    state.values["breadth_state_v2"] = compute_breadth_v2_features(
        sector_etf_closes=sector_closes,
        config=state.breadth_state_v2_config,
        pit_constituent_intervals=state.context.pit_constituent_intervals,
        constituent_ohlcv=state.context.constituent_ohlcv,
    )


def _build_volume_liquidity_v2_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.volume_liquidity_v2_config is None
        or "volume" not in state.spy_ohlcv.columns
        or state.spy_ohlcv["volume"].isna().all()
    ):
        state.values["volume_liquidity_v2"] = None
        return
    state.values["volume_liquidity_v2"] = compute_volume_liquidity_v2_features(
        volume=state.spy_ohlcv["volume"],
        config=state.volume_liquidity_v2_config,
    )


def _build_monetary_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.monetary_pressure_v2_config is None
        or state.context.macro_series is None
        or _FRED_DGS2_KEY not in state.context.macro_series
        or _FRED_DGS10_KEY not in state.context.macro_series
    ):
        state.values["monetary"] = None
        return
    broad_usd_series = state.context.macro_series.get("broad_usd_index")
    cb_text_score_series: pd.Series | None = None
    if (
        state.central_bank_text_config is not None
        and state.context.central_bank_text_releases is not None
        and not state.context.central_bank_text_releases.empty
    ):
        cb_text_score_series = to_daily_score_series(
            state.context.central_bank_text_releases,
            session_index=state.spy_close.index,
            smoothing_window_sessions=state.central_bank_text_config.smoothing_window_sessions,
            same_date_aggregation=state.central_bank_text_config.same_date_aggregation,
        )
    state.values["monetary"] = compute_monetary_pressure_features(
        dgs2=state.context.macro_series[_FRED_DGS2_KEY],
        dgs10=state.context.macro_series[_FRED_DGS10_KEY],
        broad_usd_index=broad_usd_series,
        central_bank_text_score=cb_text_score_series,
        config=state.monetary_pressure_v2_config,
    )


def _build_realized_vol_21d_feature(state: _FeatureStoreBuildState) -> None:
    state.values["realized_vol_21d"] = (
        realized_vol(state.spy_close, 21)
        if (
            state.context.config.hmm is not None
            or state.context.config.clustering is not None
            or state.context.config.change_point is not None
        )
        else None
    )


def _build_hmm_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.context.config.hmm is None
        or state.values["volume_liquidity_v2"] is None
        or state.values["network_fragility"] is None
    ):
        state.values["hmm"] = None
        return
    state.values["hmm"] = compute_hmm_features(
        return_1d=state.values["volatility"].return_1d,
        realized_vol_21d=state.values["realized_vol_21d"],
        drawdown_63d=compute_trailing_drawdown(state.spy_close, 63),
        volume_zscore_20d=state.values["volume_liquidity_v2"].volume_zscore_20d,
        avg_pairwise_corr_63d=state.values["network_fragility"].avg_pairwise_corr_63d,
        config=state.context.config.hmm,
    )


def _build_clustering_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.context.config.clustering is None
        or state.values["breadth_state_v2"] is None
        or state.values["breadth_state_v2"].pct_above_50dma is None
        or state.values["network_fragility"] is None
        or state.values["trend_direction_v2"] is None
    ):
        state.values["clustering"] = None
        return
    state.values["clustering"] = compute_clustering_features(
        return_21d=state.values["trend_character"].return_21d,
        return_63d=state.values["trend_direction_v2"].return_63d,
        realized_vol_21d=state.values["realized_vol_21d"],
        drawdown_63d=compute_trailing_drawdown(state.spy_close, 63),
        adx_14=state.values["trend_character"].adx_14,
        avg_pairwise_corr_63d=state.values["network_fragility"].avg_pairwise_corr_63d,
        pct_above_50dma=state.values["breadth_state_v2"].pct_above_50dma,
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
        state.values["credit_funding"] = None
        return
    nan_oas = pd.Series(float("nan"), index=state.spy_close.index)
    state.values["credit_funding"] = compute_credit_funding_features(
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
    )


def _build_inflation_growth_feature(state: _FeatureStoreBuildState) -> None:
    if (
        state.inflation_growth_config is None
        or state.context.cross_asset_closes is None
        or state.context.macro_series is None
        or not all(k in state.context.cross_asset_closes for k in _IG_CROSS_ASSET_KEYS)
        or not all(k in state.context.macro_series for k in _IG_MACRO_KEYS)
    ):
        state.values["inflation_growth"] = None
        return
    state.values["inflation_growth"] = compute_inflation_growth_features(
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
        cpi_nowcast=state.context.macro_series.get("cpi_nowcast"),
        aggregate_forward_eps_revision=state.context.macro_series.get(
            "aggregate_forward_eps_revision"
        ),
        cpi_first_release=state.context.cpi_first_release,
        use_first_release_cpi_when_available=(
            state.inflation_growth_config.rules.use_first_release_cpi_when_available
        ),
    )


def _build_change_point_feature(state: _FeatureStoreBuildState) -> None:
    if state.context.config.change_point is None:
        state.values["change_point"] = None
        return
    state.values["change_point"] = compute_change_point_features(
        realized_vol_21d=state.values["realized_vol_21d"],
        config=state.context.config.change_point,
    )


_FEATURE_STORE_BUILDERS: tuple[_FeatureStoreBuilder, ...] = (
    _FeatureStoreBuilder("trend_direction", _build_trend_direction_feature),
    _FeatureStoreBuilder("sentiment_score", _build_sentiment_score_feature),
    _FeatureStoreBuilder("news_sentiment_score", _build_news_sentiment_score_feature),
    _FeatureStoreBuilder("trend_direction_v2", _build_trend_direction_v2_feature),
    _FeatureStoreBuilder("trend_character", _build_trend_character_feature),
    _FeatureStoreBuilder("volatility", _build_volatility_feature),
    _FeatureStoreBuilder("breadth", _build_breadth_feature),
    _FeatureStoreBuilder("sma_50", _build_sma_50_feature),
    _FeatureStoreBuilder("network_fragility", _build_network_fragility_feature),
    _FeatureStoreBuilder("volatility_state_v2", _build_volatility_state_v2_feature),
    _FeatureStoreBuilder("breadth_state_v2", _build_breadth_state_v2_feature),
    _FeatureStoreBuilder("volume_liquidity_v2", _build_volume_liquidity_v2_feature),
    _FeatureStoreBuilder("monetary", _build_monetary_feature),
    _FeatureStoreBuilder("realized_vol_21d", _build_realized_vol_21d_feature),
    _FeatureStoreBuilder("hmm", _build_hmm_feature),
    _FeatureStoreBuilder("clustering", _build_clustering_feature),
    _FeatureStoreBuilder("credit_funding", _build_credit_funding_feature),
    _FeatureStoreBuilder("inflation_growth", _build_inflation_growth_feature),
    _FeatureStoreBuilder("change_point", _build_change_point_feature),
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
) -> FeatureStore:
    # TODO(refactor): Decompose this builder in a dedicated no-behavior-change
    # refactor. Keep feature wiring and fixture replay frozen while extracting
    # helpers so classifier changes do not hide inside the decomposition.
    spy_ohlcv = context.spy_ohlcv
    spy_close = spy_ohlcv["close"]
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
    _run_feature_store_builders(_FEATURE_STORE_BUILDERS, build_state)
    values = build_state.values

    return FeatureStore(
        spy_index=spy_ohlcv.index,
        trend_direction=values["trend_direction"],
        trend_character=values["trend_character"],
        volatility=values["volatility"],
        breadth=values["breadth"],
        sma_50=values["sma_50"],
        network_fragility=values["network_fragility"],
        trend_direction_v2=values["trend_direction_v2"],
        volatility_state_v2=values["volatility_state_v2"],
        breadth_state_v2=values["breadth_state_v2"],
        volume_liquidity_v2=values["volume_liquidity_v2"],
        monetary=values["monetary"],
        hmm=values["hmm"],
        clustering=values["clustering"],
        change_point=values["change_point"],
        credit_funding=values["credit_funding"],
        inflation_growth=values["inflation_growth"],
    )
