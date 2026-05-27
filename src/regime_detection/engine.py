from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from regime_detection.calendar import as_date, require_nyse_trading_day
from regime_detection.config import (
    RegimeConfig,
    load_default_regime_config,
    load_regime_config,
)
from regime_detection.market_context import build_market_context
from regime_detection.models import RegimeOutput, RegimeTimeline
from regime_detection.timeline import build_regime_timeline


@dataclass(frozen=True)
class ClassifyRequest:
    end_date: date
    market_data: pd.DataFrame
    lookback_days: int = 1
    breadth_data: pd.DataFrame | None = None
    vix_data: pd.DataFrame | None = None
    event_calendar: pd.DataFrame | None = None
    config: RegimeConfig | None = None
    sector_etf_closes: dict[str, pd.Series] | None = None
    cross_asset_closes: dict[str, pd.Series] | None = None
    macro_series: dict[str, pd.Series] | None = None
    pit_constituent_intervals: pd.DataFrame | None = None
    constituent_ohlcv: dict[str, pd.DataFrame] | None = None
    aaii_sentiment: pd.DataFrame | None = None
    implied_vol_30d: pd.Series | None = None
    central_bank_text_releases: pd.DataFrame | None = None
    cpi_first_release: pd.Series | None = None
    news_sentiment: pd.Series | None = None
    breadth_data_compatibility: Literal["ignored_legacy_parameter"] = (
        "ignored_legacy_parameter"
    )


def _ignore_legacy_breadth_data(breadth_data: pd.DataFrame | None) -> None:
    """Preserve the public V1 parameter while current breadth uses RSP/SPY rows."""
    _ = breadth_data


def _require_event_calendar(event_calendar: pd.DataFrame | None) -> pd.DataFrame:
    if event_calendar is None:
        raise ValueError(
            "event_calendar is required for RegimeEngine classification. "
            "Pass the manifest event_calendar DataFrame from the runner/caller."
        )
    if not isinstance(event_calendar, pd.DataFrame):
        raise TypeError(
            "event_calendar must be a pandas DataFrame when passed to RegimeEngine."
        )
    return event_calendar


class RegimeEngine:
    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            self._config = load_default_regime_config()
        else:
            self._config = load_regime_config(Path(config_path))

    @property
    def config(self) -> RegimeConfig:
        return self._config

    def classify(
        self,
        as_of_date: date,
        market_data: pd.DataFrame,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
        sector_etf_closes: dict[str, pd.Series] | None = None,
        cross_asset_closes: dict[str, pd.Series] | None = None,
        macro_series: dict[str, pd.Series] | None = None,
        pit_constituent_intervals: pd.DataFrame | None = None,
        constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
        aaii_sentiment: pd.DataFrame | None = None,
        implied_vol_30d: pd.Series | None = None,
        central_bank_text_releases: pd.DataFrame | None = None,
        cpi_first_release: pd.Series | None = None,
        news_sentiment: pd.Series | None = None,
    ) -> RegimeOutput:
        timeline = self.classify_request(
            ClassifyRequest(
                end_date=as_of_date,
                market_data=market_data,
                lookback_days=1,
                breadth_data=breadth_data,
                vix_data=vix_data,
                event_calendar=event_calendar,
                config=config,
                sector_etf_closes=sector_etf_closes,
                cross_asset_closes=cross_asset_closes,
                macro_series=macro_series,
                pit_constituent_intervals=pit_constituent_intervals,
                constituent_ohlcv=constituent_ohlcv,
                aaii_sentiment=aaii_sentiment,
                implied_vol_30d=implied_vol_30d,
                central_bank_text_releases=central_bank_text_releases,
                cpi_first_release=cpi_first_release,
                news_sentiment=news_sentiment,
            )
        )
        return timeline.outputs[-1]

    def classify_request(self, request: ClassifyRequest) -> RegimeTimeline:
        _ignore_legacy_breadth_data(request.breadth_data)
        end_date = as_date(request.end_date)
        require_nyse_trading_day(end_date)
        event_calendar = _require_event_calendar(request.event_calendar)
        cfg = request.config if request.config is not None else self._config
        context = build_market_context(
            end_date=end_date,
            market_data=request.market_data,
            config=cfg,
            vix_data=request.vix_data,
            event_calendar=event_calendar,
            sector_etf_closes=request.sector_etf_closes,
            cross_asset_closes=request.cross_asset_closes,
            macro_series=request.macro_series,
            pit_constituent_intervals=request.pit_constituent_intervals,
            constituent_ohlcv=request.constituent_ohlcv,
            aaii_sentiment=request.aaii_sentiment,
            implied_vol_30d=request.implied_vol_30d,
            central_bank_text_releases=request.central_bank_text_releases,
            cpi_first_release=request.cpi_first_release,
            news_sentiment=request.news_sentiment,
        )
        return build_regime_timeline(
            context=context, lookback_days=request.lookback_days, config=cfg
        )

    def classify_window(
        self,
        end_date: date,
        market_data: pd.DataFrame,
        lookback_days: int,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
        sector_etf_closes: dict[str, pd.Series] | None = None,
        cross_asset_closes: dict[str, pd.Series] | None = None,
        macro_series: dict[str, pd.Series] | None = None,
        pit_constituent_intervals: pd.DataFrame | None = None,
        constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
        aaii_sentiment: pd.DataFrame | None = None,
        implied_vol_30d: pd.Series | None = None,
        central_bank_text_releases: pd.DataFrame | None = None,
        cpi_first_release: pd.Series | None = None,
        news_sentiment: pd.Series | None = None,
    ) -> RegimeTimeline:
        return self.classify_request(
            ClassifyRequest(
                end_date=end_date,
                market_data=market_data,
                lookback_days=lookback_days,
                breadth_data=breadth_data,
                vix_data=vix_data,
                event_calendar=event_calendar,
                config=config,
                sector_etf_closes=sector_etf_closes,
                cross_asset_closes=cross_asset_closes,
                macro_series=macro_series,
                pit_constituent_intervals=pit_constituent_intervals,
                constituent_ohlcv=constituent_ohlcv,
                aaii_sentiment=aaii_sentiment,
                implied_vol_30d=implied_vol_30d,
                central_bank_text_releases=central_bank_text_releases,
                cpi_first_release=cpi_first_release,
                news_sentiment=news_sentiment,
            )
        )
