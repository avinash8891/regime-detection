from __future__ import annotations

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
from regime_detection.config import (
    BreadthV2Config,
    CreditFundingConfig,
    InflationGrowthConfig,
    MonetaryPressureV2FeaturesConfig,
    NetworkFragilityConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
    VolumeLiquidityV2Config,
)
from regime_detection.credit_funding import (
    CreditFundingFeatures,
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
    # macro_series carries SOFR/IORB/NFCI/broad_usd_index. Otherwise None
    # (V1 byte-identity preserved: RegimeOutput.credit_funding_state defaults
    # to None and is omitted on the JSON wire via exclude_none=True).
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
) -> FeatureStore:
    spy_ohlcv = context.spy_ohlcv
    spy_close = spy_ohlcv["close"]
    trend_direction = compute_trend_direction_features(spy_close)
    trend_character = compute_trend_character_features(
        close=spy_close,
        high=spy_ohlcv["high"],
        low=spy_ohlcv["low"],
        volume=spy_ohlcv["volume"] if "volume" in spy_ohlcv.columns else None,
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
        # §1A euphoria seam (Log #32 closure / ADR 0004): when context
        # carries AAII sentiment rows, derive the forward-filled
        # bull_bear_spread_8w_ma onto the SPY session index per V1 §2.2
        # stateless-replay (use the latest publication-date <= as_of_date).
        sentiment_series = _build_sentiment_score_series(
            aaii_sentiment=context.aaii_sentiment,
            session_index=spy_close.index,
        )
        trend_direction_v2 = compute_trend_v2_features(
            spy_close,
            config=trend_direction_v2_config,
            sentiment_score=sentiment_series,
        )
    else:
        trend_direction_v2 = None

    # V2 §1C volatility features (slice 2.2 + slice 2.6 rising_vol RV +
    # vol_crush IV inputs, ADR 0005 / Log #19+#20).
    if volatility_state_v2_config is not None:
        # §1C vol_crush seam (ADR 0005): when context carries the FRED
        # VIXCLS implied-vol series, pass it as implied_vol_30d so the
        # IV-derived features (implied_vol_5d_change, iv_rv_spread) are
        # computed; when absent, those features stay None and vol_crush
        # falsifies (V1 byte-identity preserved). event_window_just_passed
        # is computed from the context's event calendar when present.
        event_window = (
            compute_event_window_just_passed(
                normalized_event_calendar=context.normalized_event_calendar,
                sessions=tuple(spy_close.index.date),
                trailing_sessions=(
                    volatility_state_v2_config.rules.vol_crush_event_window_trailing_sessions
                ),
            )
            if context.normalized_event_calendar is not None
            else None
        )
        # Pass the rules sub-block so the slice-2.6 `rising_vol` rule's
        # realized_vol_short / realized_vol_long windows are populated from
        # config (rather than the hardcoded 10/63 fallback). The two paths
        # produce identical series when yaml carries the spec defaults.
        volatility_state_v2 = compute_volatility_v2_features(
            open_=spy_ohlcv["open"],
            high=spy_ohlcv["high"],
            low=spy_ohlcv["low"],
            close=spy_close,
            config=volatility_state_v2_config,
            rules_config=volatility_state_v2_config.rules,
            implied_vol_30d=context.implied_vol_30d,
            event_window_just_passed=event_window,
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
                pit_constituent_intervals=context.pit_constituent_intervals,
                constituent_ohlcv=context.constituent_ohlcv,
            )
        else:
            breadth_state_v2 = None
    else:
        breadth_state_v2 = None

    # V2 §1E volume/liquidity feature (slice 2.4) — evidence-only compute.
    # Reads SPY volume from the existing MarketContext.spy_ohlcv frame (the
    # V1 contract already requires the volume column — see
    # market_context._require_market_data_contract). Falls back to None when
    # the config is absent OR when the volume column is empty/missing.
    if (
        volume_liquidity_v2_config is not None
        and "volume" in spy_ohlcv.columns
        and not spy_ohlcv["volume"].isna().all()
    ):
        volume_liquidity_v2 = compute_volume_liquidity_v2_features(
            volume=spy_ohlcv["volume"],
            config=volume_liquidity_v2_config,
        )
    else:
        volume_liquidity_v2 = None

    # V2 §2A monetary-pressure features (slice 4.1) — evidence-only compute.
    # Requires BOTH DGS2 and DGS10 on context.macro_series (spec source
    # contract §2A lines 887–889). When either FRED key is absent OR the
    # config is None, fall back to None so v1-only callers see no diff.
    if (
        monetary_pressure_v2_config is not None
        and context.macro_series is not None
        and _FRED_DGS2_KEY in context.macro_series
        and _FRED_DGS10_KEY in context.macro_series
    ):
        # Ambiguity Log #46 (a): broad_usd_index is OPTIONAL — when absent the
        # broad_usd_index_zscore_63d output is an all-NaN series and the §2A
        # rule predicate naturally falsifies on NaN.
        broad_usd_series = None
        if context.macro_series is not None:
            broad_usd_series = context.macro_series.get("broad_usd_index")
        monetary = compute_monetary_pressure_features(
            dgs2=context.macro_series[_FRED_DGS2_KEY],
            dgs10=context.macro_series[_FRED_DGS10_KEY],
            broad_usd_index=broad_usd_series,
            config=monetary_pressure_v2_config,
        )
    else:
        monetary = None

    # v2 §6.1 HMM evidence layer (Slice 6) — reuses the existing FeatureStore
    # seams as inputs. Requires the volume_liquidity_v2 and network_fragility
    # seams to be lit (their fields supply 2 of the 5 HMM inputs). The
    # SPY-derived inputs (return_1d, realized_vol_21d, drawdown_63d) are
    # always available on the V1 path. When any predicate fails, the HMM
    # seam is None and the transition_score falls back to its 5-component
    # weights_without_hmm path (V1 byte-identity preserved).
    if (
        context.config.hmm is not None
        and volume_liquidity_v2 is not None
        and network_fragility is not None
    ):
        hmm = compute_hmm_features(
            return_1d=volatility.return_1d,
            realized_vol_21d=realized_vol(spy_close, 21),
            drawdown_63d=compute_trailing_drawdown(spy_close, 63),
            volume_zscore_20d=volume_liquidity_v2.volume_zscore_20d,
            avg_pairwise_corr_63d=network_fragility.avg_pairwise_corr_63d,
            config=context.config.hmm,
        )
    else:
        hmm = None

    # v2 §6.2 GMM clustering evidence layer (Slice 7) — diagnostic only;
    # NOT consumed by transition_score. Predicate gates on the PIT-aware
    # pct_above_50dma being lit AND network_fragility / trend_direction_v2
    # seams being populated.
    if (
        context.config.clustering is not None
        and breadth_state_v2 is not None
        and breadth_state_v2.pct_above_50dma is not None
        and network_fragility is not None
        and trend_direction_v2 is not None
    ):
        clustering = compute_clustering_features(
            return_21d=trend_character.return_21d,
            return_63d=trend_direction_v2.return_63d,
            realized_vol_21d=realized_vol(spy_close, 21),
            drawdown_63d=compute_trailing_drawdown(spy_close, 63),
            adx_14=trend_character.adx_14,
            avg_pairwise_corr_63d=network_fragility.avg_pairwise_corr_63d,
            pct_above_50dma=breadth_state_v2.pct_above_50dma,
            config=context.config.clustering,
        )
    else:
        clustering = None

    # v2 §2C credit/funding feature compute (Slice 4). Requires the eight
    # spec-pinned input series: HYG/LQD/TLT/KRE on cross_asset_closes plus
    # SOFR/IORB/NFCI/broad_usd_index on macro_series. When any input is
    # absent OR the config is None, fall back to None — V1 byte-identity
    # preserved on the credit_funding_state wire field.
    credit_funding: CreditFundingFeatures | None = None
    if (
        credit_funding_config is not None
        and context.cross_asset_closes is not None
        and context.macro_series is not None
        and all(k in context.cross_asset_closes for k in _CF_CROSS_ASSET_KEYS)
        and all(k in context.macro_series for k in _CF_MACRO_KEYS)
    ):
        # §2C credit-spread metric — single source: FRED-redistributed
        # ICE BofA OAS series (BAMLH0A0HYM2 / BAMLC0A4CBBB). Their
        # macro_series keys are in `_CF_MACRO_KEYS`, so the gate above
        # already guarantees they are present — no `.get()` needed.
        credit_funding = compute_credit_funding_features(
            tlt_close=context.cross_asset_closes[_CF_TLT_KEY],
            kre_close=context.cross_asset_closes[_CF_KRE_KEY],
            spy_close=spy_close,
            sofr=context.macro_series[_CF_SOFR_KEY],
            iorb=context.macro_series[_CF_IORB_KEY],
            nfci_weekly=context.macro_series[_CF_NFCI_KEY],
            broad_usd_index=context.macro_series[_CF_BROAD_USD_KEY],
            hy_oas=context.macro_series[_CF_HY_OAS_KEY],
            ig_oas=context.macro_series[_CF_IG_OAS_KEY],
            config=credit_funding_config.rules,
        )

    # v2 §2B inflation/growth feature compute (Slice 5). Requires all 9
    # spec-pinned input series: CPI/PMI/DGS10 on macro_series + DBC/TLT plus
    # XLY/XLI/XLP/XLU on cross_asset_closes. When any input is absent OR the
    # config is None, fall back to None — V1 byte-identity preserved on the
    # inflation_growth_state wire field.
    inflation_growth: InflationGrowthFeatures | None = None
    if (
        inflation_growth_config is not None
        and context.cross_asset_closes is not None
        and context.macro_series is not None
        and all(k in context.cross_asset_closes for k in _IG_CROSS_ASSET_KEYS)
        and all(k in context.macro_series for k in _IG_MACRO_KEYS)
    ):
        inflation_growth = compute_inflation_growth_features(
            cpi_all_items=context.macro_series[_IG_CPI_KEY],
            pmi_manufacturing=context.macro_series[_IG_PMI_KEY],
            dgs10=context.macro_series[_IG_DGS10_KEY],
            dbc_close=context.cross_asset_closes[_IG_DBC_KEY],
            spy_close=spy_close,
            tlt_close=context.cross_asset_closes[_IG_TLT_KEY],
            xly_close=context.cross_asset_closes[_IG_XLY_KEY],
            xli_close=context.cross_asset_closes[_IG_XLI_KEY],
            xlp_close=context.cross_asset_closes[_IG_XLP_KEY],
            xlu_close=context.cross_asset_closes[_IG_XLU_KEY],
            config=inflation_growth_config.rules,
        )

    # v2 §6.3 BOCPD change-point evidence layer (Slice 8). Observation
    # series is realized_vol_21d (Ambiguity Log #63) — derived from
    # SPY close on the V1 path, so the seam is only None when the config
    # is absent or when SPY history is too short.
    if context.config.change_point is not None:
        realized_vol_21d_series = realized_vol(spy_close, 21)
        change_point = compute_change_point_features(
            realized_vol_21d=realized_vol_21d_series,
            config=context.config.change_point,
        )
    else:
        change_point = None

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
        volume_liquidity_v2=volume_liquidity_v2,
        monetary=monetary,
        hmm=hmm,
        clustering=clustering,
        change_point=change_point,
        credit_funding=credit_funding,
        inflation_growth=inflation_growth,
    )
