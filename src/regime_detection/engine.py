from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.calendar import as_date, require_nyse_trading_day
from regime_detection.config import RegimeConfig, load_default_regime_config, load_regime_config
from regime_detection.market_context import build_market_context
from regime_detection.models import RegimeOutput, RegimeTimeline
from regime_detection.timeline import build_regime_timeline


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
        raise TypeError("event_calendar must be a pandas DataFrame when passed to RegimeEngine.")
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
        # TODO(api, owner=regime-maintainers): Consider a ClassifyRequest input object only if this public signature keeps growing, and ship it with a compatibility plan.
        # TODO(api, owner=regime-maintainers): Decide the deprecation path for the public V1 `breadth_data` parameter.
        _ignore_legacy_breadth_data(breadth_data)
        as_of_date = as_date(as_of_date)
        require_nyse_trading_day(as_of_date)
        timeline = self.classify_window(
            end_date=as_of_date,
            market_data=market_data,
            lookback_days=1,
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
        return timeline.outputs[-1]

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
        _ignore_legacy_breadth_data(breadth_data)
        end_date = as_date(end_date)
        require_nyse_trading_day(end_date)
        event_calendar = _require_event_calendar(event_calendar)
        cfg = config if config is not None else self._config
        context = build_market_context(
            end_date=end_date,
            market_data=market_data,
            config=cfg,
            vix_data=vix_data,
            event_calendar=event_calendar,
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
        return build_regime_timeline(
            context=context, lookback_days=lookback_days, config=cfg
        )
