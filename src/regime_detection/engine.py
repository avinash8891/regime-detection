from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.calendar import as_date
from regime_detection.config import RegimeConfig, load_default_regime_config, load_regime_config
from regime_detection.market_context import MarketContext, build_market_context, slice_context_to_end_date
from regime_detection.models import RegimeOutput, RegimeTimeline
from regime_detection.timeline import ENGINE_MINIMUM_HISTORY, build_regime_timeline


class RegimeEngine:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        if config_path is None:
            self._config = load_default_regime_config()
        else:
            self._config = load_regime_config(Path(config_path))

    @property
    def config(self) -> RegimeConfig:
        return self._config

    def classify(
        self,
        *,
        as_of_date: date,
        market_data: pd.DataFrame | None = None,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
        context: MarketContext | None = None,
    ) -> RegimeOutput:
        as_of_date = as_date(as_of_date)
        timeline = self.classify_window(
            end_date=as_of_date,
            lookback_days=ENGINE_MINIMUM_HISTORY,
            market_data=market_data,
            breadth_data=breadth_data,
            vix_data=vix_data,
            event_calendar=event_calendar,
            config=config,
            context=context,
        )
        return timeline.outputs[-1]

    def classify_window(
        self,
        *,
        end_date: date,
        lookback_days: int,
        market_data: pd.DataFrame | None = None,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
        context: MarketContext | None = None,
    ) -> RegimeTimeline:
        del breadth_data  # V1 breadth is derived from the validated market_data ETF proxy inputs.
        resolved_context = self._resolve_context(
            end_date=end_date,
            market_data=market_data,
            vix_data=vix_data,
            event_calendar=event_calendar,
            config=config,
            context=context,
        )
        return build_regime_timeline(context=resolved_context, lookback_days=lookback_days)

    def _resolve_context(
        self,
        *,
        end_date: date,
        market_data: pd.DataFrame | None,
        vix_data: pd.DataFrame | None,
        event_calendar: pd.DataFrame | None,
        config: RegimeConfig | None,
        context: MarketContext | None,
    ) -> MarketContext:
        end_date = as_date(end_date)
        if context is not None:
            if market_data is not None or vix_data is not None or event_calendar is not None:
                raise ValueError("Provide either precomputed context or raw inputs, not both.")
            if config is not None and context.config != config:
                raise ValueError("Provided context config does not match the explicit config override.")
            return slice_context_to_end_date(context=context, end_date=end_date)
        if market_data is None:
            raise ValueError("market_data is required when context is not provided.")
        cfg = config if config is not None else self._config
        return build_market_context(
            end_date=end_date,
            market_data=market_data,
            config=cfg,
            vix_data=vix_data,
            event_calendar=event_calendar,
        )
