from __future__ import annotations

from dataclasses import dataclass, field
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

ClassifyRequestSource = Literal["direct", "profile_manifest"]
_PROFILE_MANIFEST_REQUIRED_INPUTS = frozenset({"event_calendar"})


def _empty_manifest_inputs() -> frozenset[str]:
    return frozenset()


@dataclass(frozen=True)
class ClassifyRequest:
    """Single validated request object for one regime-classification run.

    The engine entry points are thin compatibility wrappers over this contract.
    Required inputs fail loudly here; optional V2 seams stay explicit as None.
    Legacy `breadth_data` is intentionally absent from the API surface.
    """

    end_date: date
    market_data: pd.DataFrame
    lookback_days: int = 1
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
    request_source: ClassifyRequestSource = "direct"
    manifest_resolved_inputs: frozenset[str] = field(
        default_factory=_empty_manifest_inputs
    )
    manifest_cli_overrides: frozenset[str] = field(
        default_factory=_empty_manifest_inputs
    )


def _require_positive_lookback_days(lookback_days: int) -> int:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    return lookback_days


def _require_event_calendar(event_calendar: object | None) -> pd.DataFrame:
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


def _validate_request_source(request: ClassifyRequest) -> None:
    """Validate runner provenance before building market context.

    Profile-manifest calls must identify the manifest-backed required inputs so
    operator diagnostics can distinguish direct calls from reproducible runner
    invocations. Direct calls may not carry manifest metadata.
    """

    manifest_inputs = request.manifest_resolved_inputs | request.manifest_cli_overrides
    if request.request_source == "direct":
        if manifest_inputs:
            raise ValueError(
                "manifest metadata requires profile_manifest request_source"
            )
        return
    if request.request_source == "profile_manifest":
        missing = _PROFILE_MANIFEST_REQUIRED_INPUTS - manifest_inputs
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "profile_manifest request missing manifest-backed required inputs: "
                f"{missing_text}"
            )
        return
    raise ValueError(
        f"unknown ClassifyRequest request_source: {request.request_source!r}"
    )


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
        request_source: ClassifyRequestSource = "direct",
        manifest_resolved_inputs: frozenset[str] | None = None,
        manifest_cli_overrides: frozenset[str] | None = None,
    ) -> RegimeOutput:
        timeline = self.classify_request(
            ClassifyRequest(
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
                request_source=request_source,
                manifest_resolved_inputs=manifest_resolved_inputs or frozenset(),
                manifest_cli_overrides=manifest_cli_overrides or frozenset(),
            )
        )
        return timeline.outputs[-1]

    def classify_request(self, request: ClassifyRequest) -> RegimeTimeline:
        """Classify a validated request through the canonical engine path."""

        _validate_request_source(request)
        lookback_days = _require_positive_lookback_days(request.lookback_days)
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
            context=context, lookback_days=lookback_days, config=cfg
        )

    def classify_window(
        self,
        end_date: date,
        market_data: pd.DataFrame,
        lookback_days: int,
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
        request_source: ClassifyRequestSource = "direct",
        manifest_resolved_inputs: frozenset[str] | None = None,
        manifest_cli_overrides: frozenset[str] | None = None,
    ) -> RegimeTimeline:
        return self.classify_request(
            ClassifyRequest(
                end_date=end_date,
                market_data=market_data,
                lookback_days=lookback_days,
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
                request_source=request_source,
                manifest_resolved_inputs=manifest_resolved_inputs or frozenset(),
                manifest_cli_overrides=manifest_cli_overrides or frozenset(),
            )
        )
