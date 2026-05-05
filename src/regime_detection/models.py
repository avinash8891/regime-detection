from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DataQualityStatus = Literal["ok", "degraded", "insufficient_data", "insufficient_history", "stale_data"]


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatus
    freshness_days: int | None
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None


class AxisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: dict[str, Any]
    data_quality: DataQuality


class BreadthStateOutput(AxisOutput):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["etf_proxy"]


class EventCalendarOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: dict[str, Any]


class MonetaryPressureOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    reason: str


class StructuralCausalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_calendar: EventCalendarOutput
    monetary_pressure: MonetaryPressureOutput


class NetworkFragilityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    reason: str


class TransitionRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: dict[str, Any]


class StrategyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_size_multiplier: float
    allow_trend_following: bool
    allow_mean_reversion: bool
    leverage_allowed: bool
    allow_buy_dip: bool
    allow_breakout: bool
    allow_shorts: bool
    require_confirmation_for_new_longs: bool
    require_confirmation_for_shorts: bool
    log_for_review: bool
    modifiers_applied: list[str]


class RegimeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    config_version: str
    as_of_date: date
    market: str

    trend_direction: AxisOutput
    trend_character: AxisOutput
    volatility_state: AxisOutput
    breadth_state: BreadthStateOutput
    structural_causal_state: StructuralCausalState
    network_fragility: NetworkFragilityOutput
    transition_risk: TransitionRiskOutput
    strategy_response: StrategyResponse
