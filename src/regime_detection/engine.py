from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from regime_detection.calendar import require_nyse_trading_day
from regime_detection.config import RegimeConfig, default_config_path, load_regime_config
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    MonetaryPressureOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskOutput,
)
from regime_detection.versioning import engine_version


class RegimeEngine:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else default_config_path()
        self._config = load_regime_config(path)

    @property
    def config(self) -> RegimeConfig:
        return self._config

    def classify(
        self,
        *,
        as_of_date: date,
        market_data: pd.DataFrame,
        breadth_data: pd.DataFrame | None = None,
        vix_data: pd.DataFrame | None = None,
        event_calendar: pd.DataFrame | None = None,
        config: RegimeConfig | None = None,
    ) -> RegimeOutput:
        cfg = config if config is not None else self._config
        if cfg.trading_calendar != "NYSE":
            raise ValueError(f"V1 supports only NYSE trading calendar. Got: {cfg.trading_calendar}")

        require_nyse_trading_day(as_of_date)

        _require_market_data_contract(market_data, as_of_date=as_of_date)

        # Slice 1 (Foundation) only: emit unknown/not_implemented labels for
        # classifier axes until subsequent slices implement them.
        unknown_axis = _unknown_axis_output()
        unknown_breadth = _unknown_breadth_output()

        structural = StructuralCausalState(
            event_calendar=_unknown_event_calendar_output(),
            monetary_pressure=MonetaryPressureOutput(
                label="unknown",
                reason="not_implemented_v1",
            ),
        )

        return RegimeOutput(
            engine_version=engine_version(),
            config_version=cfg.config_version,
            as_of_date=as_of_date,
            market="SPY",
            trend_direction=unknown_axis,
            trend_character=unknown_axis,
            volatility_state=unknown_axis,
            breadth_state=unknown_breadth,
            structural_causal_state=structural,
            network_fragility=NetworkFragilityOutput(
                label="not_implemented_v1",
                reason="breadth_state_used_as_v1_fragility_proxy",
            ),
            transition_risk=TransitionRiskOutput(
                label="stable",
                evidence={"warnings_active": []},
            ),
            strategy_response=_default_strategy_response(),
        )


def _require_market_data_contract(df: pd.DataFrame, *, as_of_date: date) -> None:
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    if df.empty:
        raise ValueError("market_data must not be empty")
    if (df["symbol"] == "SPY").sum() == 0:
        raise ValueError("market_data must contain SPY rows for V1")
    dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    has_spy_asof = ((df["symbol"] == "SPY") & (dates == as_of_date)).any()
    if not bool(has_spy_asof):
        raise ValueError(f"market_data must include SPY row for as_of_date={as_of_date.isoformat()}")


def _unknown_data_quality(*, reason: str, status: str) -> DataQuality:
    return DataQuality(
        status=status, freshness_days=None, completeness=None, reason=reason
    )


def _unknown_axis_output() -> AxisOutput:
    return AxisOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "not_implemented_v1"},
        data_quality=_unknown_data_quality(reason="not_implemented_v1", status="insufficient_data"),
    )


def _unknown_breadth_output() -> BreadthStateOutput:
    return BreadthStateOutput(
        mode="etf_proxy",
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"reason": "not_implemented_v1", "proxy": "RSP/SPY"},
        data_quality=_unknown_data_quality(reason="not_implemented_v1", status="insufficient_data"),
    )


def _unknown_event_calendar_output() -> EventCalendarOutput:
    return EventCalendarOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={"all_matching_events": [], "selected_via_precedence": "unknown"},
    )


def _default_strategy_response() -> StrategyResponse:
    # Slice 1 placeholder: strategy_response is fully defined in Slice 9.
    return StrategyResponse(
        position_size_multiplier=1.0,
        allow_trend_following=True,
        allow_mean_reversion=True,
        leverage_allowed=True,
        allow_buy_dip=True,
        allow_breakout=True,
        allow_shorts=True,
        require_confirmation_for_new_longs=False,
        require_confirmation_for_shorts=False,
        log_for_review=True,
        modifiers_applied=["not_implemented_v1"],
    )
