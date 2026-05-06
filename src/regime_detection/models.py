from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DataQualityStatus = Literal["ok", "degraded", "insufficient_data", "insufficient_history", "stale_data"]


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatus
    freshness_days: int | None = Field(default=None, ge=0)
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

class StructuralCausalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_calendar: EventCalendarOutput


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

    # V1 modifier fields (conditionally emitted; omit when not applicable).
    hard_max_loss_required: bool | None = None
    block_weak_signals: bool | None = None
    prefer_cash_or_hedges: bool | None = None
    take_profit_faster: bool | None = None
    allow_leverage_expansion: bool | None = None
    require_breadth_confirmation: bool | None = None
    reason: str | None = None


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
    transition_risk: TransitionRiskOutput
    strategy_response: StrategyResponse
