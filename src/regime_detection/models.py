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
    reason: str | None = None


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


class LabelReasonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    reason: str


class NetworkFragilityOutput(BaseModel):
    """Layer 3 network fragility classifier output (v2 spec §3).

    Until slice 1 ships the v2 fragility classifier, emit `unknown` labels
    with `data_quality.status="insufficient_history"` per v1 §2.7 NaN
    handling pattern.
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: dict[str, Any]
    data_quality: DataQuality
    mode: Literal["sector_cross_asset_22"] = "sector_cross_asset_22"


class MonetaryPressureOutput(BaseModel):
    """Monetary pressure classifier output (v2 spec §2A).

    Until slice 4 ships the v2 monetary-pressure classifier, emit
    `label="unknown"` with `data_quality.status="insufficient_history"`.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: dict[str, Any]
    data_quality: DataQuality


class InflationGrowthOutput(BaseModel):
    """Inflation/growth state output (v2 spec §2B). Minimal until slice 5."""

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: dict[str, Any]
    data_quality: DataQuality


CreditFundingLabel = Literal[
    "credit_calm",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


class CreditFundingOutput(BaseModel):
    """v2 §2C credit/funding state output (Slice 4).

    Three-tier label triple (raw/stable/active) per the v2 axis pattern.
    ``evidence`` carries the per-day scalar rule inputs and the bias-warning
    code (``credit_spread_proxy_total_return_differential``).
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: CreditFundingLabel
    stable_label: CreditFundingLabel
    active_label: CreditFundingLabel
    evidence: dict[str, Any]
    data_quality: DataQuality


class VolumeLiquidityOutput(BaseModel):
    """Volume / liquidity internals output (v2 spec §1E). Minimal until slice 2."""

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: dict[str, Any]
    data_quality: DataQuality


VolumeLiquidityLabel = Literal[
    "normal_volume",
    "panic_volume",
    "liquidity_gap_behavior",
    "unknown",
]


class VolumeLiquidityStateOutput(BaseModel):
    """v2 §1E volume/liquidity state output (Slice 2.7).

    Carries the three-tier label triple (raw/stable/active) the v2
    axes use, plus per-day evidence and a data-quality record. The
    ``mode`` literal pins the only compute path shipped today: a
    z-score over SPY's daily share volume (`volume_zscore_v1`). When
    the feature seam is None (no volume column) the timeline emits a
    placeholder via the engine wiring rather than instantiating this
    output class.
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: VolumeLiquidityLabel
    stable_label: VolumeLiquidityLabel
    active_label: VolumeLiquidityLabel
    evidence: dict[str, Any]
    data_quality: DataQuality
    mode: Literal["volume_zscore_v1"] = "volume_zscore_v1"


class ClusterOutput(BaseModel):
    """v2 §6.2 clustering output (Slice 7). Diagnostic evidence; per-day
    cluster assignment + Mahalanobis distance to the assigned-cluster
    centroid. ``mapped_label`` is omitted until the operator-curated
    ``cluster_label_map.yaml`` ships (spec line 2842 + V2 §10).
    """

    model_config = ConfigDict(extra="forbid")

    cluster_id: int  # raw 0..n_clusters-1; NOT an economic label
    distance_to_centroid: float
    model_version: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class ChangePointOutput(BaseModel):
    """Change-point detection output (v2 spec §4.6 — V2.1 ship)."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    days_since_last_break: int = Field(ge=0)
    evidence: dict[str, Any]


class StructuralCausalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_calendar: EventCalendarOutput
    monetary_pressure: MonetaryPressureOutput


TransitionScoreInterpretation = Literal[
    "stable", "weakening", "transition_warning", "high"
]


class TransitionRiskOutput(BaseModel):
    """Layer 4 transition risk output.

    V1 emits `label` + `evidence` (named warnings per v1 §9). V2 §4 adds a
    continuous composite `score` and its components; these are optional
    until slice 3 ships the v2 transition-score composer.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: dict[str, Any]

    # V2 §4.5 transition score augments (does not replace) V1 named warnings.
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_interpretation: TransitionScoreInterpretation | None = None
    score_components: dict[str, float] | None = None


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
    """v2 §5.2 — resolved per-family constraint shape (Slice 5.2).

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


class AgentRouting(BaseModel):
    """v2 §5.1 Agent Cohort Routing output (Slice 5.1)."""

    model_config = ConfigDict(extra="forbid")

    active_cohort: str
    fallback_cohort: str
    blocked_cohorts: list[str]


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

    # V2 optional top-level fields (default None → omitted from wire via
    # exclude_none=True). Each lands when its v2 slice ships.
    inflation_growth_state: InflationGrowthOutput | None = None  # v2 §2B (slice 5)
    credit_funding_state: CreditFundingOutput | None = None  # v2 §2C (slice 4)
    volume_liquidity_state: VolumeLiquidityStateOutput | None = None  # v2 §1E (slice 2.7)
    change_point: ChangePointOutput | None = None  # v2 §4.6 (V2.1)
    cluster: ClusterOutput | None = None  # v2 §6.2 (Slice 7) — diagnostic evidence
    agent_routing: "AgentRouting | None" = None  # v2 §5.1 (slice 5.1)
    strategy_family_constraints: dict[str, StrategyFamilyConstraint] | None = None  # v2 §5.2 (slice 5.2)

    # V1 wire contract: omit any None-valued conditional fields in nested models.
    # This must be applied at the top-level dump to propagate into nested models.
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class RegimeTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    config_version: str
    market: str
    start_date: date
    end_date: date
    trading_calendar: str
    outputs: list[RegimeOutput]
