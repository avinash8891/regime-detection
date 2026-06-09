from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from regime_detection._rolling_stats import simple_moving_average
from regime_detection.breadth_state import (
    BreadthFeatures,
    compute_features as compute_breadth_features,
)
from regime_detection.breadth_state import (
    BreadthV2Features,
    compute_breadth_v2_features,
)
from regime_detection.change_point import (
    ChangePointConfig,
    ChangePointFeatures,
    compute_change_point_features,
)
from regime_detection.central_bank_text import to_daily_score_series
from regime_detection.config import (
    BreadthV2Config,
    CentralBankTextConfig,
    CreditFundingConfig,
    CreditFundingRulesConfig,
    InflationGrowthConfig,
    InflationGrowthRulesConfig,
    MonetaryPressureV2FeaturesConfig,
    NetworkFragilityConfig,
    NewsSentimentConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
    VolumeLiquidityV2Config,
)
from regime_detection.credit_funding import (
    CreditFundingFeatures,
    compute_credit_funding_features,
)
from regime_detection.credit_funding_rules import (
    BROAD_USD_INDEX_KEY as _CF_BROAD_USD_KEY,
    FEDFUNDS_KEY as _CF_FEDFUNDS_KEY,
    HYG_KEY as _CF_HYG_KEY,
    HY_OAS_KEY as _CF_HY_OAS_KEY,
    IG_OAS_KEY as _CF_IG_OAS_KEY,
    IOER_LEGACY_KEY as _CF_IOER_LEGACY_KEY,
    IORB_KEY as _CF_IORB_KEY,
    KRE_KEY as _CF_KRE_KEY,
    LQD_KEY as _CF_LQD_KEY,
    NFCI_KEY as _CF_NFCI_KEY,
    REQUIRED_CROSS_ASSET_KEYS as _CF_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as _CF_MACRO_KEYS,
    SOFR_KEY as _CF_SOFR_KEY,
    TLT_KEY as _CF_TLT_KEY,
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
from regime_detection.trend_direction import (
    TrendDirectionV2Features,
    compute_trailing_drawdown,
    compute_trend_v2_features,
)
from regime_detection.clustering import (
    ClusteringConfig,
    ClusteringFeatures,
    compute_clustering_features,
)
from regime_detection.hmm_state import HMMConfig, HMMFeatures, compute_hmm_features
from regime_detection.volatility_state import (
    VolatilityFeatures,
    VolatilityV2Features,
    compute_features as compute_volatility_features,
    compute_volatility_v2_features,
    realized_vol,
)
from regime_detection.volume_liquidity import (
    VolumeLiquidityV2Features,
    compute_volume_liquidity_v2_features,
)
from regime_detection.monetary_pressure import (
    MonetaryPressureV2Features,
    compute_monetary_pressure_features,
)
from regime_detection.inflation_growth import (
    InflationGrowthFeatures,
    compute_inflation_growth_features,
)
from regime_detection.inflation_growth_rules import (
    AGG_FORWARD_EPS_REVISION_KEY as _IG_AGG_FORWARD_EPS_REVISION_KEY,
    CPI_KEY as _IG_CPI_KEY,
    CPI_NOWCAST_KEY as _IG_CPI_NOWCAST_KEY,
    DBC_KEY as _IG_DBC_KEY,
    DGS10_KEY as _IG_DGS10_KEY,
    PMI_KEY as _IG_PMI_KEY,
    REQUIRED_CROSS_ASSET_KEYS as _IG_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as _IG_MACRO_KEYS,
    TLT_KEY as _IG_TLT_KEY,
    XLI_KEY as _IG_XLI_KEY,
    XLP_KEY as _IG_XLP_KEY,
    XLU_KEY as _IG_XLU_KEY,
    XLY_KEY as _IG_XLY_KEY,
)
from regime_detection.feature_store_runtime import (
    FeatureAvailability,
    FeatureSpec,
    _Unavailable,
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
# v2 §1A (spec lines 231-233): euphoria's sentiment_score is NaN until at least
# this many weekly AAII readings exist on or before the session (F-006).
_MIN_SENTIMENT_WEEKLY_READINGS = 4
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
    consult a future-dated reading. Returns ``None`` when no AAII frame is
    supplied (lets the euphoria predicate falsify per the V2 §10 absolute
    "do not invent" rule at spec L4364).

    Cold-start (v2 §1A spec lines 231-233 / ADR 0004 Q5): ``sentiment_score``
    is NaN until at least ``_MIN_SENTIMENT_WEEKLY_READINGS`` (4) weekly readings
    exist on or before the session; the euphoria predicate then falsifies on those
    sessions per V1 §2.7. The AAII fetcher's `min_periods=1` 8-week MA otherwise
    exposes an under-warmed value from week 1, which would let euphoria fire on
    only 1-3 readings (F-006) — so the under-4-reading window is masked here.

    Note on the Optional contract: the FeatureSpec resolver in
    `_resolve_sentiment_score` gates the None/empty/missing-column cases
    before calling this helper, so the orchestrator path always passes
    valid input. The Optional return is preserved here for direct external
    callers (e.g. unit tests) that exercise the no-AAII path.
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
    sorted_aaii = (
        # kind="mergesort" is a STABLE sort: rows sharing a publication_date keep their
        # source order, so the keep="last" dedupe below deterministically retains the
        # same physical row every replay. The default quicksort is unstable and could
        # pick a different duplicate row across runs/pandas versions, breaking the
        # byte-identical replay contract.
        aaii_sentiment.sort_values(publication_column, kind="mergesort")
        # CR-008: collapse duplicate publication dates so the warmup counts DISTINCT
        # weekly readings (a re-published row must not warm early) and the ffill reindex
        # below stays on a unique index.
        .drop_duplicates(subset=[publication_column], keep="last").reset_index(
            drop=True
        )
    )
    publication = pd.DatetimeIndex(pd.to_datetime(sorted_aaii[publication_column]))
    score_values = sorted_aaii["bull_bear_spread_8w_ma"].astype(float).to_numpy()
    aligned = pd.Series(
        score_values,
        index=publication,
        name="sentiment_score",
    )
    result = aligned.reindex(session_index, method="ffill")
    # v2 §1A cold-start (spec lines 231-233): sentiment_score is NaN until at least 4
    # weekly readings exist on or before the session. `publication` is now distinct
    # (deduped above, CR-008), so this counts DISTINCT weekly publication dates; the
    # under-warmed warmup window is masked to NaN so the euphoria predicate falsifies
    # instead of firing on 1-3 readings (F-006).
    readings_on_or_before = np.asarray(
        # pandas-stubs types DatetimeIndex.searchsorted as -> Unknown; np.asarray with
        # an explicit dtype gives a concrete int array for the warm-count comparison.
        publication.searchsorted(session_index, side="right"),  # type: ignore[reportUnknownMemberType]
        dtype=int,
    )
    warm = pd.Series(
        readings_on_or_before >= _MIN_SENTIMENT_WEEKLY_READINGS,
        index=session_index,
    )
    return result.where(warm)


def _build_news_sentiment_score_series(
    *,
    news_sentiment: pd.Series,
    session_index: pd.DatetimeIndex,
    config: NewsSentimentConfig,
) -> pd.Series:
    score = (
        news_sentiment.reindex(session_index, method="ffill")
        .rolling(config.smoothing_window_sessions, min_periods=1)
        .mean()
    )
    score.name = "news_sentiment_score"
    return score


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


def _build_sentiment_score(
    aaii_sentiment: pd.DataFrame, session_index: pd.DatetimeIndex
) -> pd.Series | None:
    # _build_sentiment_score_series declares an Optional return for external
    # callers (see helper docstring); when invoked via the spec, resolve has
    # already gated the None/empty/missing-column cases, so this returns a
    # populated Series in practice. The Optional return matches the helper's
    # signature so pyright stays clean without a cast.
    return _build_sentiment_score_series(
        aaii_sentiment=aaii_sentiment, session_index=session_index
    )


def _resolve_sentiment_score(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    aaii = state.context.aaii_sentiment
    if aaii is None or aaii.empty:
        return _Unavailable(missing_inputs=("aaii_sentiment",))
    if "bull_bear_spread_8w_ma" not in aaii.columns:
        return _Unavailable(missing_inputs=("aaii_sentiment.bull_bear_spread_8w_ma",))
    if "publication_date" not in aaii.columns and "date" not in aaii.columns:
        return _Unavailable(missing_inputs=("aaii_sentiment.publication_date_or_date",))
    return {
        "aaii_sentiment": aaii,
        "session_index": _as_datetime_index(state.spy_close.index),
    }


def _build_news_sentiment_score(
    news_sentiment: pd.Series,
    session_index: pd.DatetimeIndex,
    config: NewsSentimentConfig,
) -> pd.Series:
    return _build_news_sentiment_score_series(
        news_sentiment=news_sentiment, session_index=session_index, config=config
    )


def _resolve_news_sentiment_score(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.news_sentiment_config is None:
        missing.append("news_sentiment_config")
    if state.context.news_sentiment is None or state.context.news_sentiment.empty:
        missing.append("news_sentiment")
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    return {
        "news_sentiment": state.context.news_sentiment,
        "session_index": _as_datetime_index(state.spy_close.index),
        "config": state.news_sentiment_config,
    }


def _build_trend_direction_v2(
    spy_close: pd.Series,
    config: TrendDirectionV2Config,
    sentiment_score: pd.Series | None,
    news_sentiment_score: pd.Series | None,
) -> TrendDirectionV2Features:
    return compute_trend_v2_features(
        spy_close,
        config=config,
        sentiment_score=sentiment_score,
        news_sentiment_score=news_sentiment_score,
    )


def _resolve_trend_direction_v2(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    if state.trend_direction_v2_config is None:
        return _Unavailable(missing_inputs=("trend_direction_v2_config",))
    return {
        "spy_close": state.spy_close,
        "config": state.trend_direction_v2_config,
        "sentiment_score": state.sentiment_score,
        "news_sentiment_score": state.news_sentiment_score,
    }


def _build_network_fragility(
    sector_etf_closes: dict[str, pd.Series],
    cross_asset_closes: dict[str, pd.Series],
    spy_close: pd.Series,
    config: NetworkFragilityConfig | None,
) -> NetworkFragilityFeatures:
    if config is None:
        return compute_network_fragility_features(
            sector_etf_closes=sector_etf_closes,
            cross_asset_closes=cross_asset_closes,
            spy_close=spy_close,
        )
    return compute_network_fragility_features(
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        spy_close=spy_close,
        correlation_lookback_days=config.correlation_lookback_days,
        percentile_lookback_days=config.percentile_lookback_days,
        realized_vol_lookback_days=config.realized_vol_lookback_days,
        dispersion_percentile_lookback_days=config.dispersion_percentile_lookback_days,
        dispersion_spy_vol_floor=config.dispersion_spy_vol_floor,
        min_universe_size=config.min_universe_size,
        min_window_completeness=config.min_window_completeness,
        universe=config.universe,
    )


def _resolve_network_fragility(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    if state.context.sector_etf_closes is None:
        return _Unavailable(missing_inputs=("sector_etf_closes",))
    return {
        "sector_etf_closes": state.context.sector_etf_closes,
        "cross_asset_closes": state.context.cross_asset_closes or {},
        "spy_close": state.spy_close,
        "config": state.network_fragility_config,
    }


def _build_volatility_state_v2(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: VolatilityV2Config,
    implied_vol_30d: pd.Series | None,
    event_window_just_passed: pd.Series | None,
) -> VolatilityV2Features:
    return compute_volatility_v2_features(
        open_=open_,
        high=high,
        low=low,
        close=close,
        config=config,
        rules_config=config.rules,
        implied_vol_30d=implied_vol_30d,
        event_window_just_passed=event_window_just_passed,
    )


def _resolve_volatility_state_v2(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    if state.volatility_state_v2_config is None:
        return _Unavailable(missing_inputs=("volatility_state_v2_config",))
    config = state.volatility_state_v2_config
    event_window = (
        compute_event_window_just_passed(
            normalized_event_calendar=state.context.normalized_event_calendar,
            sessions=tuple(
                ts.date() for ts in _as_datetime_index(state.spy_close.index)
            ),
            trailing_sessions=config.rules.vol_crush_event_window_trailing_sessions,
        )
        if state.context.normalized_event_calendar is not None
        else None
    )
    return {
        "open_": _series_column(state.spy_ohlcv, "open"),
        "high": _series_column(state.spy_ohlcv, "high"),
        "low": _series_column(state.spy_ohlcv, "low"),
        "close": state.spy_close,
        "config": config,
        "implied_vol_30d": state.context.implied_vol_30d,
        "event_window_just_passed": event_window,
    }


def _build_breadth_state_v2(
    sector_etf_closes: dict[str, pd.Series],
    config: BreadthV2Config,
    pit_constituent_intervals: pd.DataFrame | None,
    constituent_ohlcv: dict[str, pd.DataFrame] | None,
) -> BreadthV2Features:
    return compute_breadth_v2_features(
        sector_etf_closes=sector_etf_closes,
        config=config,
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
    )


def _resolve_breadth_state_v2(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.breadth_state_v2_config is None:
        missing.append("breadth_state_v2_config")
    # Sector_etf_closes missingness uses the same helper the legacy report uses,
    # which returns ("sector_etf_closes",) when None or
    # ("sector_etf_closes.any_sector_etf",) when none of SECTOR_ETFS match.
    missing.extend(_missing_sector_inputs(state))
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    sector_closes = state.context.sector_etf_closes
    assert sector_closes is not None  # narrowed by _missing_sector_inputs check
    return {
        "sector_etf_closes": sector_closes,
        "config": state.breadth_state_v2_config,
        "pit_constituent_intervals": state.context.pit_constituent_intervals,
        "constituent_ohlcv": state.context.constituent_ohlcv,
    }


def _build_volume_liquidity_v2(
    volume: pd.Series,
    config: VolumeLiquidityV2Config,
) -> VolumeLiquidityV2Features:
    return compute_volume_liquidity_v2_features(volume=volume, config=config)


def _resolve_volume_liquidity_v2(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.volume_liquidity_v2_config is None:
        missing.append("volume_liquidity_v2_config")
    spy_volume: pd.Series | None = None
    if "volume" not in state.spy_ohlcv.columns:
        missing.append("spy_ohlcv.volume")
    else:
        spy_volume = _series_column(state.spy_ohlcv, "volume")
        if bool(spy_volume.isna().all()):
            missing.append("spy_ohlcv.volume.non_nan")
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    assert spy_volume is not None  # narrowed by the missing check above
    return {
        "volume": spy_volume,
        "config": state.volume_liquidity_v2_config,
    }


def _build_monetary(
    dgs2: pd.Series,
    dgs10: pd.Series,
    broad_usd_index: pd.Series,
    central_bank_text_score: pd.Series | None,
    config: MonetaryPressureV2FeaturesConfig,
) -> MonetaryPressureV2Features:
    return compute_monetary_pressure_features(
        dgs2=dgs2,
        dgs10=dgs10,
        broad_usd_index=broad_usd_index,
        central_bank_text_score=central_bank_text_score,
        config=config,
    )


def _build_realized_vol_21d(spy_close: pd.Series) -> pd.Series:
    return realized_vol(spy_close, 21)


def _build_drawdown_63d(spy_close: pd.Series) -> pd.Series:
    return compute_trailing_drawdown(spy_close, 63)


def _resolve_drawdown_63d(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    """Disjunction gate: drawdown_63d is built when hmm OR clustering config
    is present (those features consume it). When both are None, _Unavailable —
    matches legacy 'state.drawdown_63d = None' branch."""
    if state.context.config.hmm is None and state.context.config.clustering is None:
        return _Unavailable(missing_inputs=("hmm_or_clustering_config",))
    return {"spy_close": state.spy_close}


def _build_hmm(
    config: HMMConfig,
    return_1d: pd.Series,
    realized_vol_21d: pd.Series,
    drawdown_63d: pd.Series,
    volume_zscore_20d: pd.Series,
    avg_pairwise_corr_63d: pd.Series,
) -> HMMFeatures | None:
    return compute_hmm_features(
        return_1d=return_1d,
        realized_vol_21d=realized_vol_21d,
        drawdown_63d=drawdown_63d,
        volume_zscore_20d=volume_zscore_20d,
        avg_pairwise_corr_63d=avg_pairwise_corr_63d,
        config=config,
    )


def _resolve_hmm(
    state: _FeatureStoreBuildState,
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
    assert state.volume_liquidity_v2 is not None
    assert state.network_fragility is not None
    assert state.realized_vol_21d is not None
    assert state.drawdown_63d is not None
    return {
        "config": state.context.config.hmm,
        "return_1d": volatility.return_1d,
        "realized_vol_21d": state.realized_vol_21d,
        "drawdown_63d": state.drawdown_63d,
        "volume_zscore_20d": state.volume_liquidity_v2.volume_zscore_20d,
        "avg_pairwise_corr_63d": state.network_fragility.avg_pairwise_corr_63d,
    }


def _build_clustering(
    config: ClusteringConfig,
    return_21d: pd.Series,
    return_63d: pd.Series,
    realized_vol_21d: pd.Series,
    drawdown_63d: pd.Series,
    adx_14: pd.Series,
    avg_pairwise_corr_63d: pd.Series,
    pct_above_50dma: pd.Series,
) -> ClusteringFeatures | None:
    return compute_clustering_features(
        return_21d=return_21d,
        return_63d=return_63d,
        realized_vol_21d=realized_vol_21d,
        drawdown_63d=drawdown_63d,
        adx_14=adx_14,
        avg_pairwise_corr_63d=avg_pairwise_corr_63d,
        pct_above_50dma=pct_above_50dma,
        config=config,
    )


def _resolve_clustering(
    state: _FeatureStoreBuildState,
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
    assert state.trend_direction_v2 is not None
    assert state.network_fragility is not None
    assert state.realized_vol_21d is not None
    assert state.drawdown_63d is not None
    assert breadth_state_v2 is not None
    assert breadth_state_v2.pct_above_50dma is not None
    return {
        "config": state.context.config.clustering,
        "return_21d": trend_character.return_21d,
        "return_63d": state.trend_direction_v2.return_63d,
        "realized_vol_21d": state.realized_vol_21d,
        "drawdown_63d": state.drawdown_63d,
        "adx_14": trend_character.adx_14,
        "avg_pairwise_corr_63d": state.network_fragility.avg_pairwise_corr_63d,
        "pct_above_50dma": breadth_state_v2.pct_above_50dma,
    }


def _build_credit_funding(
    hyg_close: pd.Series,
    lqd_close: pd.Series,
    tlt_close: pd.Series,
    kre_close: pd.Series,
    spy_close: pd.Series,
    sofr: pd.Series,
    iorb: pd.Series,
    nfci_weekly: pd.Series,
    broad_usd_index: pd.Series,
    hy_oas: pd.Series,
    ig_oas: pd.Series,
    config: CreditFundingRulesConfig,
    fedfunds: pd.Series | None,
    ioer_legacy: pd.Series | None,
) -> CreditFundingFeatures:
    return compute_credit_funding_features(
        hyg_close=hyg_close,
        lqd_close=lqd_close,
        tlt_close=tlt_close,
        kre_close=kre_close,
        spy_close=spy_close,
        sofr=sofr,
        iorb=iorb,
        nfci_weekly=nfci_weekly,
        broad_usd_index=broad_usd_index,
        hy_oas=hy_oas,
        ig_oas=ig_oas,
        config=config,
        fedfunds=fedfunds,
        ioer_legacy=ioer_legacy,
    )


def _resolve_credit_funding(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.credit_funding_config is None:
        missing.append("credit_funding_config")
        return _Unavailable(missing_inputs=tuple(missing))
    cross_missing = _missing_cross_asset_keys(
        state.context.cross_asset_closes, tuple(_CF_CROSS_ASSET_KEYS)
    )
    macro_missing = _missing_macro_keys(
        state.context.macro_series, tuple(_CF_MACRO_KEYS)
    )
    missing.extend(cross_missing)
    missing.extend(macro_missing)
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    assert state.context.cross_asset_closes is not None
    assert state.context.macro_series is not None
    nan_oas = pd.Series(float("nan"), index=state.spy_close.index)
    return {
        "hyg_close": state.context.cross_asset_closes[_CF_HYG_KEY],
        "lqd_close": state.context.cross_asset_closes[_CF_LQD_KEY],
        "tlt_close": state.context.cross_asset_closes[_CF_TLT_KEY],
        "kre_close": state.context.cross_asset_closes[_CF_KRE_KEY],
        "spy_close": state.spy_close,
        "sofr": state.context.macro_series[_CF_SOFR_KEY],
        "iorb": state.context.macro_series[_CF_IORB_KEY],
        "nfci_weekly": state.context.macro_series[_CF_NFCI_KEY],
        "broad_usd_index": state.context.macro_series[_CF_BROAD_USD_KEY],
        "hy_oas": state.context.macro_series.get(_CF_HY_OAS_KEY, nan_oas),
        "ig_oas": state.context.macro_series.get(_CF_IG_OAS_KEY, nan_oas),
        "config": state.credit_funding_config.rules,
        "fedfunds": state.context.macro_series.get(_CF_FEDFUNDS_KEY),
        "ioer_legacy": state.context.macro_series.get(_CF_IOER_LEGACY_KEY),
    }


def _build_inflation_growth(
    cpi_all_items: pd.Series,
    pmi_manufacturing: pd.Series,
    dgs10: pd.Series,
    dbc_close: pd.Series,
    spy_close: pd.Series,
    tlt_close: pd.Series,
    xly_close: pd.Series,
    xli_close: pd.Series,
    xlp_close: pd.Series,
    xlu_close: pd.Series,
    config: InflationGrowthRulesConfig,
    cpi_nowcast: pd.Series | None,
    aggregate_forward_eps_revision: pd.Series | None,
    cpi_first_release: pd.Series | None,
    use_first_release_cpi_when_available: bool,
) -> InflationGrowthFeatures:
    return compute_inflation_growth_features(
        cpi_all_items=cpi_all_items,
        pmi_manufacturing=pmi_manufacturing,
        dgs10=dgs10,
        dbc_close=dbc_close,
        spy_close=spy_close,
        tlt_close=tlt_close,
        xly_close=xly_close,
        xli_close=xli_close,
        xlp_close=xlp_close,
        xlu_close=xlu_close,
        config=config,
        cpi_nowcast=cpi_nowcast,
        aggregate_forward_eps_revision=aggregate_forward_eps_revision,
        cpi_first_release=cpi_first_release,
        use_first_release_cpi_when_available=use_first_release_cpi_when_available,
    )


def _resolve_inflation_growth(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.inflation_growth_config is None:
        missing.append("inflation_growth_config")
        return _Unavailable(missing_inputs=tuple(missing))
    cross_missing = _missing_cross_asset_keys(
        state.context.cross_asset_closes, tuple(_IG_CROSS_ASSET_KEYS)
    )
    macro_missing = _missing_macro_keys(
        state.context.macro_series, tuple(_IG_MACRO_KEYS)
    )
    missing.extend(cross_missing)
    missing.extend(macro_missing)
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    assert state.context.cross_asset_closes is not None
    assert state.context.macro_series is not None
    return {
        "cpi_all_items": state.context.macro_series[_IG_CPI_KEY],
        "pmi_manufacturing": state.context.macro_series[_IG_PMI_KEY],
        "dgs10": state.context.macro_series[_IG_DGS10_KEY],
        "dbc_close": state.context.cross_asset_closes[_IG_DBC_KEY],
        "spy_close": state.spy_close,
        "tlt_close": state.context.cross_asset_closes[_IG_TLT_KEY],
        "xly_close": state.context.cross_asset_closes[_IG_XLY_KEY],
        "xli_close": state.context.cross_asset_closes[_IG_XLI_KEY],
        "xlp_close": state.context.cross_asset_closes[_IG_XLP_KEY],
        "xlu_close": state.context.cross_asset_closes[_IG_XLU_KEY],
        "config": state.inflation_growth_config.rules,
        "cpi_nowcast": state.context.macro_series.get(_IG_CPI_NOWCAST_KEY),
        "aggregate_forward_eps_revision": state.context.macro_series.get(
            _IG_AGG_FORWARD_EPS_REVISION_KEY
        ),
        "cpi_first_release": state.context.cpi_first_release,
        "use_first_release_cpi_when_available": (
            state.inflation_growth_config.rules.use_first_release_cpi_when_available
        ),
    }


def _build_change_point(
    realized_vol_21d: pd.Series,
    config: ChangePointConfig,
) -> ChangePointFeatures | None:
    """compute_change_point_features can return None when BOCPD training-window
    data is insufficient. The orchestrator (Task 2.0b) emits available=False,
    reason='not_configured' for None returns — matching legacy semantics."""
    return compute_change_point_features(
        realized_vol_21d=realized_vol_21d, config=config
    )


def _resolve_change_point(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    if state.context.config.change_point is None:
        return _Unavailable(missing_inputs=("change_point_config",))
    if state.realized_vol_21d is None:
        return _Unavailable(missing_inputs=("realized_vol_21d",))
    return {
        "realized_vol_21d": state.realized_vol_21d,
        "config": state.context.config.change_point,
    }


def _resolve_realized_vol_21d(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    """Disjunction gate: realized_vol_21d is built when ANY of hmm,
    clustering, or change_point configs is present (those features consume it).
    When none are set, return _Unavailable so build is skipped — matches
    legacy 'state.realized_vol_21d = None' branch."""
    if (
        state.context.config.hmm is None
        and state.context.config.clustering is None
        and state.context.config.change_point is None
    ):
        return _Unavailable(
            missing_inputs=("hmm_or_clustering_or_change_point_config",)
        )
    return {"spy_close": state.spy_close}


def _resolve_monetary(
    state: _FeatureStoreBuildState,
) -> dict[str, object] | _Unavailable:
    if state.monetary_pressure_v2_config is None:
        # Unconfigured monetary axis is expected absence — reason="not_configured"
        # via the orchestrator's empty-missing_inputs branch, paired with the
        # spec's default policy="none" so coverage marks the run safe.
        return _Unavailable(missing_inputs=())
    macro_missing = _missing_macro_keys(
        state.context.macro_series,
        (_FRED_DGS2_KEY, _IG_DGS10_KEY, "broad_usd_index"),
    )
    if macro_missing:
        # Monetary IS configured but required macro data is missing — this is
        # an UNSAFE data gap for direct build_feature_store / build_regime_timeline
        # callers that bypass the ClassifyRequest validator. Override the spec's
        # default policy ("none", for opt-out callers) with "raise" so
        # classification_coverage flags the run as unsafe.
        return _Unavailable(
            missing_inputs=tuple(macro_missing),
            policy_override="raise",
        )
    assert state.context.macro_series is not None  # _missing_macro_keys narrowed
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
    return {
        "dgs2": state.context.macro_series[_FRED_DGS2_KEY],
        "dgs10": state.context.macro_series[_IG_DGS10_KEY],
        "broad_usd_index": state.context.macro_series["broad_usd_index"],
        "central_bank_text_score": cb_text_score_series,
        "config": state.monetary_pressure_v2_config,
    }


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
    FeatureSpec(
        name="sentiment_score",
        policy="none",
        required_inputs=("aaii_sentiment",),
        resolve=_resolve_sentiment_score,
        build=_build_sentiment_score,
        store=lambda s, v: setattr(s, "sentiment_score", v),
        report=False,
    ),
    FeatureSpec(
        name="news_sentiment_score",
        policy="none",
        required_inputs=("news_sentiment_config", "news_sentiment"),
        resolve=_resolve_news_sentiment_score,
        build=_build_news_sentiment_score,
        store=lambda s, v: setattr(s, "news_sentiment_score", v),
        report=False,
    ),
    FeatureSpec(
        name="trend_direction_v2",
        policy="none",
        required_inputs=("trend_direction_v2_config", "spy_ohlcv.close"),
        resolve=_resolve_trend_direction_v2,
        build=_build_trend_direction_v2,
        store=lambda s, v: setattr(s, "trend_direction_v2", v),
    ),
    FeatureSpec(
        name="network_fragility",
        policy="none",
        required_inputs=("sector_etf_closes",),
        resolve=_resolve_network_fragility,
        build=_build_network_fragility,
        store=lambda s, v: setattr(s, "network_fragility", v),
    ),
    FeatureSpec(
        name="volatility_state_v2",
        policy="none",
        required_inputs=("volatility_state_v2_config", "spy_ohlcv.ohlc"),
        resolve=_resolve_volatility_state_v2,
        build=_build_volatility_state_v2,
        store=lambda s, v: setattr(s, "volatility_state_v2", v),
    ),
    FeatureSpec(
        name="breadth_state_v2",
        policy="none",
        required_inputs=("breadth_state_v2_config", "sector_etf_closes"),
        resolve=_resolve_breadth_state_v2,
        build=_build_breadth_state_v2,
        store=lambda s, v: setattr(s, "breadth_state_v2", v),
    ),
    FeatureSpec(
        name="volume_liquidity_v2",
        policy="none",
        required_inputs=("volume_liquidity_v2_config", "spy_ohlcv.volume"),
        resolve=_resolve_volume_liquidity_v2,
        build=_build_volume_liquidity_v2,
        store=lambda s, v: setattr(s, "volume_liquidity_v2", v),
    ),
    FeatureSpec(
        # monetary uses policy="none" because the monetary_pressure_v2 axis is
        # optional in V2 config — when unconfigured, absence is expected and
        # downstream coverage must not flag the run as unsafe. The
        # configured-but-missing-data case is enforced upstream by the
        # ClassifyRequest input-contract validator at engine.py, which raises
        # ValueError before the feature store is built. Legacy availability
        # reported policy="raise" here unconditionally, which made
        # classification_coverage mark every V1-mode run as unsafe.
        name="monetary",
        policy="none",
        required_inputs=(
            "macro_series",
            _FRED_DGS2_KEY,
            _IG_DGS10_KEY,
            "broad_usd_index",
        ),
        resolve=_resolve_monetary,
        build=_build_monetary,
        store=lambda s, v: setattr(s, "monetary", v),
    ),
    FeatureSpec(
        name="realized_vol_21d",
        policy="none",
        required_inputs=("hmm_or_clustering_or_change_point_config",),
        resolve=_resolve_realized_vol_21d,
        build=_build_realized_vol_21d,
        store=lambda s, v: setattr(s, "realized_vol_21d", v),
        report=False,
    ),
    FeatureSpec(
        name="drawdown_63d",
        policy="none",
        required_inputs=("hmm_or_clustering_config",),
        resolve=_resolve_drawdown_63d,
        build=_build_drawdown_63d,
        store=lambda s, v: setattr(s, "drawdown_63d", v),
        report=False,
    ),
    FeatureSpec(
        name="hmm",
        policy="none",
        required_inputs=("hmm_config", "volume_liquidity_v2", "network_fragility"),
        resolve=_resolve_hmm,
        build=_build_hmm,
        store=lambda s, v: setattr(s, "hmm", v),
    ),
    FeatureSpec(
        name="clustering",
        policy="none",
        required_inputs=(
            "clustering_config",
            "breadth_state_v2.pct_above_50dma",
            "network_fragility",
            "trend_direction_v2",
        ),
        resolve=_resolve_clustering,
        build=_build_clustering,
        store=lambda s, v: setattr(s, "clustering", v),
    ),
    FeatureSpec(
        name="credit_funding",
        policy="none",
        required_inputs=("credit_funding_config", "cross_asset_closes", "macro_series"),
        resolve=_resolve_credit_funding,
        build=_build_credit_funding,  # pyright: ignore[reportUnknownArgumentType]
        store=lambda s, v: setattr(s, "credit_funding", v),
    ),
    FeatureSpec(
        name="inflation_growth",
        policy="none",
        required_inputs=(
            "inflation_growth_config",
            "cross_asset_closes",
            "macro_series",
        ),
        resolve=_resolve_inflation_growth,
        build=_build_inflation_growth,  # pyright: ignore[reportUnknownArgumentType]
        store=lambda s, v: setattr(s, "inflation_growth", v),
    ),
    FeatureSpec(
        name="change_point",
        policy="none",
        required_inputs=("change_point_config", "realized_vol_21d"),
        resolve=_resolve_change_point,
        build=_build_change_point,
        store=lambda s, v: setattr(s, "change_point", v),
    ),
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
    availability = _run_feature_specs(_FEATURE_SPECS, build_state)

    return FeatureStore(
        spy_index=_as_datetime_index(spy_ohlcv.index),
        availability=availability,
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
