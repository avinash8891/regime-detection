"""V2 market-state feature spec resolvers for the feature store."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from regime_detection.config import (
    BreadthV2Config,
    NetworkFragilityConfig,
    TrendDirectionV2Config,
    VolatilityV2Config,
    VolumeLiquidityV2Config,
)
from regime_detection.event_calendar import compute_event_window_just_passed
from regime_detection.feature_store_runtime import (
    _Unavailable,
    _require_build_input,
)
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.market_context import MarketContext


class V2MarketFeatureState(Protocol):
    context: MarketContext
    spy_ohlcv: pd.DataFrame
    spy_close: pd.Series
    network_fragility_config: NetworkFragilityConfig | None
    trend_direction_v2_config: TrendDirectionV2Config | None
    volatility_state_v2_config: VolatilityV2Config | None
    breadth_state_v2_config: BreadthV2Config | None
    volume_liquidity_v2_config: VolumeLiquidityV2Config | None
    sentiment_score: pd.Series | None
    news_sentiment_score: pd.Series | None


def resolve_trend_direction_v2(
    state: V2MarketFeatureState,
) -> dict[str, object] | _Unavailable:
    if state.trend_direction_v2_config is None:
        return _Unavailable(missing_inputs=("trend_direction_v2_config",))
    return {
        "spy_close": state.spy_close,
        "config": state.trend_direction_v2_config,
        "sentiment_score": state.sentiment_score,
        "news_sentiment_score": state.news_sentiment_score,
    }


def resolve_network_fragility(
    state: V2MarketFeatureState,
) -> dict[str, object] | _Unavailable:
    if state.context.sector_etf_closes is None:
        return _Unavailable(missing_inputs=("sector_etf_closes",))
    return {
        "sector_etf_closes": state.context.sector_etf_closes,
        "cross_asset_closes": state.context.cross_asset_closes or {},
        "spy_close": state.spy_close,
        "config": state.network_fragility_config,
    }


def resolve_volatility_state_v2(
    state: V2MarketFeatureState,
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


def resolve_breadth_state_v2(
    state: V2MarketFeatureState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.breadth_state_v2_config is None:
        missing.append("breadth_state_v2_config")
    missing.extend(missing_sector_inputs(state))
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


def resolve_volume_liquidity_v2(
    state: V2MarketFeatureState,
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


def missing_sector_inputs(state: V2MarketFeatureState) -> tuple[str, ...]:
    sector_closes = state.context.sector_etf_closes
    if sector_closes is None:
        return ("sector_etf_closes",)
    if not any(symbol in sector_closes for symbol in SECTOR_ETFS):
        return ("sector_etf_closes.any_sector_etf",)
    return ()


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise RuntimeError("feature store requires a DatetimeIndex-backed SPY frame")
    return index


def _series_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column]
