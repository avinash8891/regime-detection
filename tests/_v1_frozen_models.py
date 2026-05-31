"""Frozen V1 wire-shape models (snapshot at Phase B / commit 482e44b).

These classes mirror the intentionally pinned V1 fixture contract used by the
current test suite. The event-calendar subtree is allowed to evolve with
explicit fixture migrations because calendar state is now multi-label.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DataQualityStatusV1Frozen = Literal[
    "ok", "degraded", "insufficient_data", "insufficient_history", "stale_data"
]


class DataQualityV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatusV1Frozen
    freshness_days: int | None = Field(default=None, ge=0)
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


class AxisOutputV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: dict[str, Any]
    data_quality: DataQualityV1Frozen


class BreadthStateOutputV1Frozen(AxisOutputV1Frozen):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["etf_proxy"]


class EventCalendarOutputV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_label: str
    matching_labels: tuple[str, ...]
    evidence: dict[str, Any]


class StateReasonOutputV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    reason: str


class StructuralCausalStateV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_calendar: EventCalendarOutputV1Frozen
    monetary_pressure: StateReasonOutputV1Frozen


class TransitionRiskOutputV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    evidence: dict[str, Any]


class StrategyResponseV1Frozen(BaseModel):
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

    hard_max_loss_required: bool | None = None
    block_weak_signals: bool | None = None
    prefer_cash_or_hedges: bool | None = None
    take_profit_faster: bool | None = None
    allow_leverage_expansion: bool | None = None
    require_breadth_confirmation: bool | None = None
    reason: str | None = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class RegimeOutputV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    config_version: str
    as_of_date: date
    market: str

    trend_direction: AxisOutputV1Frozen
    trend_character: AxisOutputV1Frozen
    volatility_state: AxisOutputV1Frozen
    breadth_state: BreadthStateOutputV1Frozen
    structural_causal_state: StructuralCausalStateV1Frozen
    network_fragility: StateReasonOutputV1Frozen
    transition_risk: TransitionRiskOutputV1Frozen
    strategy_response: StrategyResponseV1Frozen

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class RegimeTimelineV1Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    config_version: str
    market: str
    start_date: date
    end_date: date
    trading_calendar: str
    outputs: list[RegimeOutputV1Frozen]
