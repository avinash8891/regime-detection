from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from regime_detection.axis_output_models import EventCalendarOutput
from regime_detection.classification_status import DataQuality
from regime_detection.evidence_payloads import TransitionRiskEvidencePayload
from regime_detection.model_status import ClassificationStatus


class StructuralCausalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_calendar: EventCalendarOutput


TransitionRiskState = Literal[
    "stable",
    "watch",
    "weakening",
    "transition_warning",
    "high_transition_risk",
    "crisis",
    "bear_stress",
    "fragile_bull",
    "recovery_attempt",
    "insufficient_data",
]


class TransitionRiskOutput(BaseModel):
    """Layer 4 transition risk output.

    `state` is the final transition-risk decision selected from data quality,
    component score, and hard-rule overrides. The score and explanation fields
    explain the decision; there is no separate legacy label path.
    """

    model_config = ConfigDict(extra="forbid")

    state: TransitionRiskState
    evidence: TransitionRiskEvidencePayload
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_components: dict[str, float] | None = None
    primary_drivers: list[str] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    data_quality: DataQuality

    # Symmetry with AxisOutput: surface a normalized classification status so
    # downstream audit/report tooling can distinguish "classified" from
    # "insufficient_history" cold-start rows without re-deriving from label.
    # When state is insufficient_data we mark the row as insufficient_history;
    # everything else is classified.
    classification_status: ClassificationStatus | None = None

    @model_validator(mode="after")
    def _populate_classification_status(self) -> "TransitionRiskOutput":
        self.classification_status = (
            "insufficient_history"
            if self.state == "insufficient_data"
            else "classified"
        )
        return self


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

    # V1 wire contract: modifier fields are omitted when not applicable.
    # Default `exclude_none=True` prevents emitting `"field": null` unless a caller opts in.
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class StrategyFamilyConstraint(BaseModel):
    """v2 §5.2 — resolved per-family constraint shape.

    Carries the post-inheritance constraint values for one strategy family
    under one active cohort. ``allowed`` is the only required dimension;
    every other field is Optional with ``None`` meaning "not specified for
    this family under this cohort" (omitted from the JSON wire via the
    overridden ``model_dump``).
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    max_lookback_days: int | None = None
    max_holding_days: int | None = None
    max_position_pct: float | None = None
    min_adx: int | None = None
    require_breadth_confirmation: bool | None = None
    require_volume_confirmation: bool | None = None
    event_window_only: bool | None = None
    reason: str | None = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class EffectiveStrategyConstraint(BaseModel):
    """Canonical per-family/mode strategy permission after all layers resolve."""

    model_config = ConfigDict(extra="forbid")

    family: str
    allowed: bool
    sources: list[str]
    blocking_reasons: list[str] = Field(default_factory=list)
    position_size_multiplier: float
    leverage_allowed: bool
    require_confirmation_for_new_longs: bool
    require_confirmation_for_shorts: bool
    max_lookback_days: int | None = None
    max_holding_days: int | None = None
    max_position_pct: float | None = None
    min_adx: int | None = None
    require_breadth_confirmation: bool | None = None
    require_volume_confirmation: bool | None = None
    event_window_only: bool | None = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class AgentRouting(BaseModel):
    """v2 §5.1 Agent Cohort Routing output.

    ``blocked_strategy_modes`` names strategy modes/families the active cohort
    suppresses; it does not list alternate agent cohorts.
    """

    model_config = ConfigDict(extra="forbid")

    active_cohort: str
    fallback_cohort: str
    blocked_strategy_modes: list[str]
