"""Feature spec definitions and build-state for the regime-detection feature store.

This module is package-private (leading underscore). It defines:
- ``_FeatureStoreBuildState``: mutable accumulator threaded through all feature specs.
- ``_FEATURE_SPECS``: the ordered tuple of ``FeatureSpec`` instances consumed by
  ``build_feature_store`` via ``_run_feature_specs``.
- Core ``_resolve_*`` / ``_build_*`` helper functions and the two sentinel-score
  series builders (``_build_sentiment_score_series``,
  ``_build_news_sentiment_score_series``). Trainable-evidence resolvers live in
  ``_feature_specs_trainable``; macro-axis resolvers live in
  ``_feature_specs_macro``.
- Module-level constants ``_FRED_DGS2_KEY``, ``_MIN_SENTIMENT_WEEKLY_READINGS``,
  that are used here and re-exposed by ``feature_store`` where needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
from regime_detection.credit_funding import (
    CreditFundingFeatures,
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
    DGS10_KEY as _IG_DGS10_KEY,
)
from regime_detection.feature_store_runtime import (
    FeatureSpec,
    _Unavailable,
    _require_build_input,
    _require_feature,
)
from regime_detection._feature_specs_trainable import (
    resolve_change_point,
    resolve_clustering,
    resolve_drawdown_63d,
    resolve_hmm,
    resolve_realized_vol_21d,
)
from regime_detection._feature_specs_macro import (
    FRED_DGS2_KEY,
    resolve_credit_funding,
    resolve_inflation_growth,
    resolve_monetary,
)

# v2 §1A (spec lines 231-233): euphoria's sentiment_score is NaN until at least
# this many weekly AAII readings exist on or before the session (F-006).
_MIN_SENTIMENT_WEEKLY_READINGS = 4


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
    sentiment_score_config: SentimentScoreConfig | None = None
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


def _missing_sector_inputs(state: _FeatureStoreBuildState) -> tuple[str, ...]:
    sector_closes = state.context.sector_etf_closes
    if sector_closes is None:
        return ("sector_etf_closes",)
    if not any(symbol in sector_closes for symbol in SECTOR_ETFS):
        return ("sector_etf_closes.any_sector_etf",)
    return ()


def _build_sentiment_score_series(
    *,
    aaii_sentiment: pd.DataFrame | None,
    session_index: pd.DatetimeIndex,
    config: SentimentScoreConfig | None = None,
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

    The FeatureSpec resolver in `_resolve_sentiment_score` gates the
    None/empty/missing-column cases before calling this helper. Direct callers
    get the same fail-loud contract instead of a silent ``None`` seam.
    """
    if aaii_sentiment is None:
        raise ValueError("aaii_sentiment is required")
    if aaii_sentiment.empty:
        raise ValueError("aaii_sentiment must not be empty")
    if "bull_bear_spread_8w_ma" not in aaii_sentiment.columns:
        raise ValueError("aaii_sentiment missing bull_bear_spread_8w_ma")
    publication_column = (
        "publication_date" if "publication_date" in aaii_sentiment.columns else "date"
    )
    if publication_column not in aaii_sentiment.columns:
        raise ValueError("aaii_sentiment missing publication_date_or_date")
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
    result = result.where(warm)

    # Max-staleness guard: NaN-out sessions whose last real AAII reading is
    # older than config.max_staleness_sessions sessions. This prevents the
    # euphoria gate from firing on arbitrarily stale forward-filled data.
    if config is not None:
        # For each session, find how many sessions ago the last real reading was.
        # `readings_on_or_before` is the count of publication dates on or before
        # each session. Where this count doesn't increase, it means no new real
        # reading arrived. We compute sessions_since_last_real as the distance
        # from each session to the latest publication date that is <= session.
        last_real_idx = np.clip(readings_on_or_before - 1, 0, len(publication) - 1)
        last_real_dates = publication[last_real_idx]
        # For sessions before ANY real reading (readings_on_or_before == 0),
        # already masked by warm; set last_real_dates to NaT so they stay NaN.
        no_readings_mask = readings_on_or_before == 0
        last_real_dates_series = pd.Series(last_real_dates, index=session_index)
        last_real_dates_series[no_readings_mask] = pd.NaT

        # Count NYSE sessions between the last real reading and each session.
        # Use searchsorted on session_index to get the positional index of each
        # date, then compute the difference.
        session_positions = np.arange(len(session_index))
        last_real_positions = np.asarray(
            session_index.searchsorted(last_real_dates_series, side="right") - 1,
            dtype=float,
        )
        last_real_positions[no_readings_mask] = np.nan
        sessions_since_last_real = session_positions - last_real_positions

        fresh = pd.Series(
            sessions_since_last_real <= config.max_staleness_sessions,
            index=session_index,
        )
        result = result.where(fresh)

    return result


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

    # Max-staleness guard: NaN-out sessions whose last real SF Fed news-sentiment
    # observation is older than config.max_staleness_sessions NYSE sessions.
    # `news_sentiment` is a daily Series indexed by date; original dates are the
    # dates where the raw series has a non-NaN value.
    real_dates = pd.DatetimeIndex(
        news_sentiment.index[news_sentiment.notna()]
    ).sort_values()
    if len(real_dates) > 0:
        # For each session, count how many real dates exist on or before it.
        readings_on_or_before = np.asarray(
            real_dates.searchsorted(session_index, side="right"),  # type: ignore[reportUnknownMemberType]
            dtype=int,
        )
        no_readings_mask = readings_on_or_before == 0
        last_real_idx = np.clip(readings_on_or_before - 1, 0, len(real_dates) - 1)
        last_real_dates = real_dates[last_real_idx]
        last_real_dates_series = pd.Series(last_real_dates, index=session_index)
        last_real_dates_series[no_readings_mask] = pd.NaT

        session_positions = np.arange(len(session_index))
        last_real_positions = np.asarray(
            session_index.searchsorted(last_real_dates_series, side="right") - 1,
            dtype=float,
        )
        last_real_positions[no_readings_mask] = np.nan
        sessions_since_last_real = session_positions - last_real_positions

        fresh = pd.Series(
            sessions_since_last_real <= config.max_staleness_sessions,
            index=session_index,
        )
        score = score.where(fresh)

    return score


def _resolve_trend_direction(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {"close": state.spy_close}


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


def _resolve_volatility(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {
        "close": state.spy_close,
        "vix_proxy_close": state.context.vix_proxy_close,
    }


def _resolve_breadth(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {
        "spy_close": state.spy_close,
        "rsp_close": state.context.rsp_close.reindex(state.spy_ohlcv.index),
    }


def _resolve_sma_50(
    state: _FeatureStoreBuildState,
) -> dict[str, object]:
    return {"series": state.spy_close, "window": 50}


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
        "config": state.sentiment_score_config,
    }


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
    sector_closes = _require_build_input(
        state.context.sector_etf_closes, "sector_etf_closes"
    )
    return {
        "sector_etf_closes": sector_closes,
        "config": state.breadth_state_v2_config,
        "pit_constituent_intervals": state.context.pit_constituent_intervals,
        "constituent_ohlcv": state.context.constituent_ohlcv,
    }


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
    spy_volume = _require_build_input(spy_volume, "spy_ohlcv.volume")
    return {
        "volume": spy_volume,
        "config": state.volume_liquidity_v2_config,
    }


def _build_realized_vol_21d(close: pd.Series, window: int) -> pd.Series:
    return realized_vol(close, window=window)


def _build_hmm_feature(
    *,
    return_1d: pd.Series | None,
    realized_vol_21d: pd.Series | None,
    drawdown_63d: pd.Series | None,
    volume_zscore_20d: pd.Series | None,
    avg_pairwise_corr_63d: pd.Series | None,
    config: HMMConfig,
) -> HMMFeatures | None:
    return compute_hmm_features(
        return_1d=return_1d,
        realized_vol_21d=realized_vol_21d,
        drawdown_63d=drawdown_63d,
        volume_zscore_20d=volume_zscore_20d,
        avg_pairwise_corr_63d=avg_pairwise_corr_63d,
        config=config,
    )


def _build_clustering_feature(
    *,
    return_21d: pd.Series | None,
    return_63d: pd.Series | None,
    realized_vol_21d: pd.Series | None,
    drawdown_63d: pd.Series | None,
    adx_14: pd.Series | None,
    avg_pairwise_corr_63d: pd.Series | None,
    pct_above_50dma: pd.Series | None,
    config: ClusteringConfig,
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


def _build_change_point_feature(
    *,
    realized_vol_21d: pd.Series | None,
    config: ChangePointConfig,
) -> ChangePointFeatures | None:
    return compute_change_point_features(
        realized_vol_21d=realized_vol_21d,
        config=config,
    )


_FEATURE_SPECS: tuple[FeatureSpec[object, _FeatureStoreBuildState], ...] = (
    FeatureSpec(
        name="trend_direction",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_trend_direction,
        build=compute_trend_direction_features,
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
        build=compute_volatility_features,
        store=lambda s, v: setattr(s, "volatility", v),
    ),
    FeatureSpec(
        name="breadth",
        policy="raise",
        required_inputs=("spy_ohlcv.close", "rsp_close"),
        resolve=_resolve_breadth,
        build=compute_breadth_features,
        store=lambda s, v: setattr(s, "breadth", v),
    ),
    FeatureSpec(
        name="sma_50",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_sma_50,
        build=simple_moving_average,
        store=lambda s, v: setattr(s, "sma_50", v),
    ),
    FeatureSpec(
        name="sentiment_score",
        policy="none",
        required_inputs=("aaii_sentiment",),
        resolve=_resolve_sentiment_score,
        build=_build_sentiment_score_series,
        store=lambda s, v: setattr(s, "sentiment_score", v),
        report=False,
    ),
    FeatureSpec(
        name="news_sentiment_score",
        policy="none",
        required_inputs=("news_sentiment_config", "news_sentiment"),
        resolve=_resolve_news_sentiment_score,
        build=_build_news_sentiment_score_series,
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
        build=compute_breadth_v2_features,
        store=lambda s, v: setattr(s, "breadth_state_v2", v),
    ),
    FeatureSpec(
        name="volume_liquidity_v2",
        policy="none",
        required_inputs=("volume_liquidity_v2_config", "spy_ohlcv.volume"),
        resolve=_resolve_volume_liquidity_v2,
        build=compute_volume_liquidity_v2_features,
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
            FRED_DGS2_KEY,
            _IG_DGS10_KEY,
            "broad_usd_index",
        ),
        resolve=resolve_monetary,
        build=compute_monetary_pressure_features,
        store=lambda s, v: setattr(s, "monetary", v),
    ),
    FeatureSpec(
        name="realized_vol_21d",
        policy="none",
        required_inputs=("hmm_or_clustering_or_change_point_config",),
        resolve=resolve_realized_vol_21d,
        build=_build_realized_vol_21d,
        store=lambda s, v: setattr(s, "realized_vol_21d", v),
        report=False,
    ),
    FeatureSpec(
        name="drawdown_63d",
        policy="none",
        required_inputs=("hmm_or_clustering_config",),
        resolve=resolve_drawdown_63d,
        build=compute_trailing_drawdown,
        store=lambda s, v: setattr(s, "drawdown_63d", v),
        report=False,
    ),
    FeatureSpec(
        name="hmm",
        policy="none",
        required_inputs=("hmm_config", "volume_liquidity_v2", "network_fragility"),
        resolve=resolve_hmm,
        build=_build_hmm_feature,
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
        resolve=resolve_clustering,
        build=_build_clustering_feature,
        store=lambda s, v: setattr(s, "clustering", v),
    ),
    FeatureSpec(
        name="credit_funding",
        policy="none",
        required_inputs=("credit_funding_config", "cross_asset_closes", "macro_series"),
        resolve=resolve_credit_funding,
        build=compute_credit_funding_features,  # pyright: ignore[reportUnknownArgumentType]
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
        resolve=resolve_inflation_growth,
        build=compute_inflation_growth_features,  # pyright: ignore[reportUnknownArgumentType]
        store=lambda s, v: setattr(s, "inflation_growth", v),
    ),
    FeatureSpec(
        name="change_point",
        policy="none",
        required_inputs=("change_point_config", "realized_vol_21d"),
        resolve=resolve_change_point,
        build=_build_change_point_feature,
        store=lambda s, v: setattr(s, "change_point", v),
    ),
)

FeatureStoreBuildState = _FeatureStoreBuildState
FEATURE_SPECS = _FEATURE_SPECS
as_datetime_index = _as_datetime_index
require_feature = _require_feature
series_column = _series_column
