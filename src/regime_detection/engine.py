from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from regime_detection.calendar import (
    as_date,
    nyse_sessions_between,
    require_nyse_trading_day,
)
from regime_detection.config import (
    RegimeConfig,
    load_default_regime_config,
    load_regime_config,
)
from regime_detection.credit_funding_rules import (
    REQUIRED_CROSS_ASSET_KEYS as CREDIT_FUNDING_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as CREDIT_FUNDING_MACRO_KEYS,
)
from regime_detection.inflation_growth_rules import (
    REQUIRED_CROSS_ASSET_KEYS as INFLATION_GROWTH_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as INFLATION_GROWTH_MACRO_KEYS,
)
from regime_detection.market_context import build_market_context
from regime_detection.models import RegimeOutput, RegimeTimeline
from regime_detection.timeline import build_regime_timeline

ClassifyRequestSource = Literal["direct", "profile_manifest"]
V2RequestInputPolicy = Literal["required", "optional_evidence"]
ManifestInputNames = Collection[str]


def _empty_manifest_inputs() -> frozenset[str]:
    return frozenset()


def _normalize_manifest_inputs(value: ManifestInputNames | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str | bytes):
        raise TypeError("manifest input names must be a collection of strings")
    return frozenset(value)


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
    manifest_resolved_inputs: ManifestInputNames = field(
        default_factory=_empty_manifest_inputs
    )
    manifest_cli_overrides: ManifestInputNames = field(
        default_factory=_empty_manifest_inputs
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_resolved_inputs",
            _normalize_manifest_inputs(self.manifest_resolved_inputs),
        )
        object.__setattr__(
            self,
            "manifest_cli_overrides",
            _normalize_manifest_inputs(self.manifest_cli_overrides),
        )


@dataclass(frozen=True)
class V2RequestInputContract:
    """Declared request-boundary source requirements for one configured V2 seam."""

    section: str
    config_path: str
    policy: V2RequestInputPolicy
    required_inputs: tuple[str, ...]
    rationale: str


V2_REQUEST_INPUT_CONTRACTS: tuple[V2RequestInputContract, ...] = (
    V2RequestInputContract(
        section="network_fragility",
        config_path="RegimeConfig.network_fragility",
        policy="required",
        required_inputs=("sector_etf_closes",),
        rationale="network fragility computes cross-sector correlation features",
    ),
    V2RequestInputContract(
        section="breadth_state_v2",
        config_path="RegimeConfig.breadth_state_v2",
        policy="required",
        required_inputs=("sector_etf_closes",),
        rationale="V2 breadth consumes sector ETF closes for sector breadth",
    ),
    V2RequestInputContract(
        section="volume_liquidity_v2",
        config_path="RegimeConfig.volume_liquidity_v2",
        policy="required",
        required_inputs=("spy_ohlcv.volume",),
        rationale="volume/liquidity computes SPY volume z-scores",
    ),
    V2RequestInputContract(
        section="monetary_pressure_v2",
        config_path="RegimeConfig.monetary_pressure_v2",
        policy="required",
        required_inputs=(
            "macro_series.2y_yield",
            "macro_series.10y_yield",
            "macro_series.broad_usd_index",
        ),
        rationale="monetary pressure features consume yield and USD macro series",
    ),
    V2RequestInputContract(
        section="monetary_pressure_state",
        config_path="RegimeConfig.monetary_pressure_state",
        policy="required",
        required_inputs=(
            "macro_series.2y_yield",
            "macro_series.10y_yield",
            "macro_series.broad_usd_index",
        ),
        rationale="monetary pressure state is configured only with usable macro features",
    ),
    V2RequestInputContract(
        section="credit_funding",
        config_path="RegimeConfig.credit_funding",
        policy="required",
        required_inputs=tuple(
            f"cross_asset_closes.{key}" for key in CREDIT_FUNDING_CROSS_ASSET_KEYS
        )
        + tuple(f"macro_series.{key}" for key in CREDIT_FUNDING_MACRO_KEYS),
        rationale="credit/funding consumes credit ETFs, bank ETF, NFCI, funding, and USD series",
    ),
    V2RequestInputContract(
        section="inflation_growth",
        config_path="RegimeConfig.inflation_growth",
        policy="required",
        required_inputs=tuple(
            f"cross_asset_closes.{key}" for key in INFLATION_GROWTH_CROSS_ASSET_KEYS
        )
        + tuple(f"macro_series.{key}" for key in INFLATION_GROWTH_MACRO_KEYS),
        rationale="inflation/growth consumes macro, commodity, rates, and sector-pair inputs",
    ),
    V2RequestInputContract(
        section="hmm",
        config_path="RegimeConfig.hmm",
        policy="required",
        required_inputs=("volume_liquidity_v2", "network_fragility"),
        rationale="HMM evidence uses volume/liquidity and network features as inputs",
    ),
    V2RequestInputContract(
        section="clustering",
        config_path="RegimeConfig.clustering",
        policy="required",
        required_inputs=(
            "breadth_state_v2",
            "network_fragility",
            "trend_direction_v2",
        ),
        rationale="clustering evidence uses breadth, network, and trend V2 features",
    ),
    V2RequestInputContract(
        section="change_point",
        config_path="RegimeConfig.change_point",
        policy="required",
        required_inputs=("spy_ohlcv.close",),
        rationale="change-point evidence consumes realized volatility from SPY close",
    ),
    V2RequestInputContract(
        section="central_bank_text",
        config_path="RegimeConfig.central_bank_text",
        policy="optional_evidence",
        required_inputs=(),
        rationale="central-bank text is evidence-only and may be absent",
    ),
    V2RequestInputContract(
        section="news_sentiment",
        config_path="RegimeConfig.news_sentiment",
        policy="optional_evidence",
        required_inputs=(),
        rationale="news sentiment is evidence-only and may be absent",
    ),
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

    manifest_inputs = _normalize_manifest_inputs(
        request.manifest_resolved_inputs
    ) | _normalize_manifest_inputs(request.manifest_cli_overrides)
    if request.request_source == "direct":
        if manifest_inputs:
            raise ValueError(
                "manifest metadata requires profile_manifest request_source"
            )
        return
    if request.request_source == "profile_manifest":
        # F-048: §2.1 requires profile_manifest calls to "pass manifest provenance"
        # but does NOT enumerate which input names must appear. Require NON-EMPTY
        # provenance rather than a hardcoded literal (the prior gate demanded
        # 'event_calendar' specifically, rejecting a runner that legitimately resolved
        # provenance under different names). Direct vs profile_manifest is still cleanly
        # distinguished — a profile_manifest run must carry at least one resolved input
        # or CLI override.
        if not manifest_inputs:
            raise ValueError(
                "profile_manifest request missing manifest provenance: "
                "manifest_resolved_inputs / manifest_cli_overrides must be non-empty"
            )
        return
    raise ValueError(
        f"unknown ClassifyRequest request_source: {request.request_source!r}"
    )


def _validate_v2_request_input_contracts(
    request: ClassifyRequest, cfg: RegimeConfig
) -> None:
    if cfg.config_version == "core3-v1.0.0":
        return
    missing_by_section: dict[str, tuple[str, ...]] = {}
    for contract in V2_REQUEST_INPUT_CONTRACTS:
        if getattr(cfg, contract.section) is None:
            continue
        if contract.policy == "optional_evidence":
            continue
        missing = tuple(
            required
            for required in contract.required_inputs
            if not _request_input_is_present(request, cfg, required)
        )
        if missing:
            missing_by_section[contract.section] = missing
    if missing_by_section:
        detail = "; ".join(
            f"{section}: {', '.join(missing)}"
            for section, missing in sorted(missing_by_section.items())
        )
        raise ValueError(f"ClassifyRequest missing configured V2 inputs: {detail}")


def _request_input_is_present(
    request: ClassifyRequest, cfg: RegimeConfig, required_input: str
) -> bool:
    if required_input.startswith("cross_asset_closes."):
        key = required_input.removeprefix("cross_asset_closes.")
        return (
            request.cross_asset_closes is not None and key in request.cross_asset_closes
        )
    if required_input.startswith("macro_series."):
        key = required_input.removeprefix("macro_series.")
        return request.macro_series is not None and key in request.macro_series
    if required_input == "sector_etf_closes":
        return bool(request.sector_etf_closes)
    if required_input == "spy_ohlcv.volume":
        return _market_data_has_non_null_spy_volume(request.market_data, cfg)
    if required_input == "spy_ohlcv.close":
        return _market_data_has_non_null_spy_close(request.market_data, cfg)
    return getattr(cfg, required_input, None) is not None


def _market_data_has_non_null_spy_volume(
    market_data: pd.DataFrame, cfg: RegimeConfig
) -> bool:
    symbol = cfg.etf_proxy.cap_weight_index
    if "symbol" not in market_data.columns or "volume" not in market_data.columns:
        return False
    volume = market_data.loc[market_data["symbol"] == symbol, "volume"]
    return not volume.empty and not bool(volume.isna().all())


def _market_data_has_non_null_spy_close(
    market_data: pd.DataFrame, cfg: RegimeConfig
) -> bool:
    symbol = cfg.etf_proxy.cap_weight_index
    if "symbol" not in market_data.columns or "close" not in market_data.columns:
        return False
    close = market_data.loc[market_data["symbol"] == symbol, "close"]
    return not close.empty and not bool(close.isna().all())


def _regime_output_to_series_row(output: RegimeOutput) -> dict[str, object]:
    """Flatten one ``RegimeOutput`` to a ``classify_series`` DataFrame row."""
    return {
        "as_of_date": output.as_of_date,
        "market": output.market,
        "engine_version": output.engine_version,
        "config_version": output.config_version,
        "trend_direction": output.trend_direction.active_label,
        "trend_character": output.trend_character.active_label,
        "volatility_state": output.volatility_state.active_label,
        "breadth_state": output.breadth_state.active_label,
        "transition_risk": output.transition_risk.state,
    }


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
        manifest_resolved_inputs: ManifestInputNames | None = None,
        manifest_cli_overrides: ManifestInputNames | None = None,
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

    def classify_series(
        self,
        start_date: date,
        end_date: date,
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
        manifest_resolved_inputs: ManifestInputNames | None = None,
        manifest_cli_overrides: ManifestInputNames | None = None,
    ) -> pd.DataFrame:
        """§2.1 V1.1 helper: per-session DataFrame of active axis labels.

        Classifies every NYSE trading session in the inclusive
        ``[start_date, end_date]`` window and returns one row per session
        (ascending by ``as_of_date``). Thin wrapper over ``classify_window`` —
        each row's labels are byte-identical to the corresponding
        ``classify(as_of_date)`` output; only the presentation (a flat frame
        instead of a ``RegimeTimeline``) differs.
        """
        start = as_date(start_date)
        end = as_date(end_date)
        if start > end:
            raise ValueError(
                f"start_date ({start.isoformat()}) must not be after end_date "
                f"({end.isoformat()})."
            )
        window_sessions = nyse_sessions_between(start, end)
        if not window_sessions:
            raise ValueError(
                "no NYSE trading sessions in "
                f"[{start.isoformat()}, {end.isoformat()}]."
            )
        timeline = self.classify_window(
            # Use the last NYSE session in the window, not the raw end_date: a
            # weekend/holiday end_date would make classify_window's
            # require_nyse_trading_day raise instead of returning the window rows.
            end_date=window_sessions[-1],
            market_data=market_data,
            lookback_days=len(window_sessions),
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
            manifest_resolved_inputs=manifest_resolved_inputs,
            manifest_cli_overrides=manifest_cli_overrides,
        )
        rows = [
            _regime_output_to_series_row(output)
            for output in timeline.outputs
            if output.as_of_date >= start
        ]
        return pd.DataFrame(rows)

    def classify_request(self, request: ClassifyRequest) -> RegimeTimeline:
        """Classify a validated request through the canonical engine path."""

        _validate_request_source(request)
        lookback_days = _require_positive_lookback_days(request.lookback_days)
        end_date = as_date(request.end_date)
        require_nyse_trading_day(end_date)
        event_calendar = _require_event_calendar(request.event_calendar)
        # F-042 / §2.4.1: a config override may only be a validated RegimeConfig
        # instance. ClassifyRequest is a plain dataclass (no Pydantic enforcement), so
        # reject a non-RegimeConfig loudly at the boundary instead of failing deep inside
        # build_market_context. The isinstance reads as "unnecessary" to pyright (the
        # static annotation is RegimeConfig | None) but it is a real RUNTIME guard — the
        # dataclass does not enforce the annotation at construction.
        if request.config is not None and not isinstance(request.config, RegimeConfig):  # type: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                "ClassifyRequest.config override must be a RegimeConfig instance "
                f"(§2.4.1), got {type(request.config).__name__}"
            )
        cfg = request.config if request.config is not None else self._config
        # F-038: validate the V2 input contracts at the boundary BEFORE building the
        # market context. A missing required V2 input must surface as the explicit
        # contract error, not as an opaque failure deep inside build_market_context.
        _validate_v2_request_input_contracts(request, cfg)
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
        manifest_resolved_inputs: ManifestInputNames | None = None,
        manifest_cli_overrides: ManifestInputNames | None = None,
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
