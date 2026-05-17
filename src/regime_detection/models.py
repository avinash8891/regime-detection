from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


DataQualityStatus = Literal["ok", "degraded", "insufficient_data", "insufficient_history", "stale_data"]
ClassificationStatus = Literal[
    "classified",
    "no_rule_fired",
    "data_unavailable",
    "stale_data",
    "insufficient_history",
    "not_wired",
]


class EvidencePayload(RootModel[dict[str, Any]]):
    """Dict-compatible named payload for unversioned regime evidence."""

    def get(self, key: str, default: Any = None) -> Any:
        return self.root.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def __iter__(self) -> Iterator[str]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def items(self) -> Any:
        return self.root.items()

    def keys(self) -> Any:
        return self.root.keys()

    def values(self) -> Any:
        return self.root.values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EvidencePayload):
            return self.root == other.root
        if isinstance(other, dict):
            return self.root == other
        return NotImplemented


class AxisEvidencePayload(EvidencePayload):
    """Dict-compatible payload for legacy V1 axis rule evidence."""


class EventCalendarEvidencePayload(EvidencePayload):
    """Dict-compatible payload for event-calendar rule evidence."""


class MonetaryPressureEvidencePayload(EvidencePayload):
    """Dict-compatible payload for monetary-pressure V2 rule evidence."""


class VolumeLiquidityEvidencePayload(EvidencePayload):
    """Dict-compatible payload for volume/liquidity V2 rule evidence."""


class TransitionRiskEvidencePayload(BaseModel):
    """Dict-compatible typed evidence payload for transition-risk warnings."""

    model_config = ConfigDict(extra="forbid")

    warnings_active: list[str]
    stable_changed_today: bool
    days_since_axis_switch: int | None

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.model_dump()[key]

    def __contains__(self, key: object) -> bool:
        return key in type(self).model_fields

    def __iter__(self) -> Iterator[str]:
        return iter(type(self).model_fields)

    def __len__(self) -> int:
        return len(type(self).model_fields)

    def items(self) -> Any:
        return self.model_dump().items()

    def keys(self) -> Any:
        return self.model_dump().keys()

    def values(self) -> Any:
        return self.model_dump().values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransitionRiskEvidencePayload):
            return self.model_dump() == other.model_dump()
        if isinstance(other, dict):
            return self.model_dump() == other
        return NotImplemented


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatus
    freshness_days: int | None = Field(default=None, ge=0)
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


def derive_classification_status(
    *,
    active_label: str,
    data_quality: DataQuality,
    evidence: EvidencePayload | None = None,
) -> tuple[ClassificationStatus, str | None]:
    """Disambiguate legacy ``unknown`` labels from data-quality failures.

    ``active_label`` remains the backward-compatible regime label. This helper
    adds the semantic reason a label was emitted, so reports can distinguish
    "data was unavailable" from "data was usable but no rule matched".
    """
    if active_label != "unknown":
        return "classified", None

    evidence_reason = None
    if evidence is not None:
        raw_reason = evidence.get("reason")
        if isinstance(raw_reason, str) and raw_reason:
            evidence_reason = raw_reason

    reason = data_quality.reason or evidence_reason
    if data_quality.status == "stale_data":
        return "stale_data", reason or "stale_data"
    if data_quality.status == "insufficient_history":
        return "insufficient_history", reason or "insufficient_history"
    if data_quality.status == "insufficient_data":
        return "data_unavailable", reason or "insufficient_data"
    return "no_rule_fired", reason or "no_rule_fired"


class AxisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: AxisEvidencePayload
    data_quality: DataQuality
    classification_status: ClassificationStatus | None = None
    classification_reason: str | None = None

    @property
    def reporting_label(self) -> str:
        if self.classification_status == "classified":
            return self.active_label
        return self.classification_status or "not_wired"

    @model_validator(mode="after")
    def _populate_classification_metadata(self) -> "AxisOutput":
        if self.classification_status is None:
            status, reason = derive_classification_status(
                active_label=self.active_label,
                data_quality=self.data_quality,
                evidence=self.evidence,
            )
            self.classification_status = status
            self.classification_reason = reason
        return self


class BreadthStateOutput(AxisOutput):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["etf_proxy", "pit_constituent_biased_research"]


class EventCalendarOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_label: str
    stable_label: str
    active_label: str
    evidence: EventCalendarEvidencePayload


class LabelReasonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    reason: str


class NetworkFragilityOutput(AxisOutput):
    """Layer 3 network fragility classifier output (v2 spec §3).

    Until slice 1 ships the v2 fragility classifier, emit `unknown` labels
    with `data_quality.status="insufficient_history"` per v1 §2.7 NaN
    handling pattern.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["sector_cross_asset_22"] = "sector_cross_asset_22"


class MonetaryPressureOutput(BaseModel):
    """Monetary pressure classifier output (v2 spec §2A).

    Until slice 4 ships the v2 monetary-pressure classifier, emit
    `label="unknown"` with `data_quality.status="insufficient_history"`.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: MonetaryPressureEvidencePayload
    data_quality: DataQuality
    classification_status: ClassificationStatus | None = None
    classification_reason: str | None = None

    @model_validator(mode="after")
    def _populate_classification_metadata(self) -> "MonetaryPressureOutput":
        if self.classification_status is None:
            status, reason = derive_classification_status(
                active_label=self.label,
                data_quality=self.data_quality,
                evidence=self.evidence,
            )
            self.classification_status = status
            self.classification_reason = reason
        return self


InflationGrowthLabel = Literal[
    "goldilocks",
    "inflation_shock",
    "disinflation",
    "recession_scare",
    "recovery_growth",
    "earnings_expansion",
    "earnings_contraction",
    "unknown",
]


class InflationGrowthOutput(AxisOutput):
    """v2 §2B inflation/growth axis output (Slice 5).

    Three-tier label triple (raw/stable/active) per the v2 axis pattern.
    ``evidence`` carries the per-day rule inputs and the bias-warning code
    (``commodity_proxy_dbc_substitute``) when applicable. The
    ``earnings_expansion``/``earnings_contraction`` labels consume the weekly
    aggregate forward-EPS revision series when it is wired and naturally
    falsify while that series is absent or in accumulator cold-start.
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: InflationGrowthLabel
    stable_label: InflationGrowthLabel
    active_label: InflationGrowthLabel


CreditFundingLabel = Literal[
    "credit_calm",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


class CreditFundingOutput(AxisOutput):
    """v2 §2C credit/funding state output (Slice 4).

    Three-tier label triple (raw/stable/active) per the v2 axis pattern.
    ``evidence`` carries the per-day scalar rule inputs and the bias-warning
    code (``credit_spread_ice_bofa_oas_fred``).
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: CreditFundingLabel
    stable_label: CreditFundingLabel
    active_label: CreditFundingLabel


MonetaryPressureV2Label = Literal[
    "tightening_pressure",
    "easing_pressure",
    "rate_shock",
    "neutral_monetary",
    "unknown",
]


class MonetaryPressureV2Output(AxisOutput):
    """v2 §2A monetary-pressure axis output (Ambiguity Log #46).

    Three-tier label triple per the v2 axis pattern (raw/stable/active);
    ``evidence`` carries the per-day scalar rule inputs; ``data_quality``
    follows the §2.8 NaN cold-start contract.
    """

    model_config = ConfigDict(extra="forbid")

    raw_label: MonetaryPressureV2Label
    stable_label: MonetaryPressureV2Label
    active_label: MonetaryPressureV2Label


class VolumeLiquidityOutput(BaseModel):
    """Volume / liquidity internals output (v2 spec §1E). Minimal until slice 2."""

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: VolumeLiquidityEvidencePayload
    data_quality: DataQuality
    classification_status: ClassificationStatus | None = None
    classification_reason: str | None = None

    @property
    def reporting_label(self) -> str:
        if self.classification_status == "classified":
            return self.label
        return self.classification_status or "not_wired"

    @model_validator(mode="after")
    def _populate_classification_metadata(self) -> "VolumeLiquidityOutput":
        if self.classification_status is None:
            status, reason = derive_classification_status(
                active_label=self.label,
                data_quality=self.data_quality,
                evidence=self.evidence,
            )
            self.classification_status = status
            self.classification_reason = reason
        return self


VolumeLiquidityLabel = Literal[
    "normal_volume",
    "panic_volume",
    "liquidity_gap_behavior",
    "unknown",
]


class VolumeLiquidityStateOutput(AxisOutput):
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
    """v2 §4.6 + §6.3 BOCPD change-point detection output (Slice 8, evidence-only).

    score: 5-session rolling max of BOCPD posterior P(run_length=0)
        (Ambiguity Log #64).
    days_since_last_break: int sessions since last posterior >= break_threshold
        (Ambiguity Log #65). None when no break has occurred in the trailing
        BOCPD window (cold-start) — omitted from the JSON wire via exclude_none.
    method: pinned to ``"BOCPD"`` for Slice 8 (Adams-MacKay 2007).
    """

    model_config = ConfigDict(extra="forbid")

    score: float
    days_since_last_break: int | None = None
    method: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


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
    evidence: TransitionRiskEvidencePayload

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
    """v2 §5.1 Agent Cohort Routing output (Slice 5.1).

    ``blocked_strategy_modes`` names strategy modes/families the active cohort
    suppresses; it does not list alternate agent cohorts.
    """

    model_config = ConfigDict(extra="forbid")

    active_cohort: str
    fallback_cohort: str
    blocked_strategy_modes: list[str]


_V1_CONFIG_VERSION = "core3-v1.0.0"


def _dump_json_payload(payload: dict[str, Any], *, indent: int | None, ensure_ascii: bool) -> str:
    json_kwargs: dict[str, Any] = {
        "ensure_ascii": ensure_ascii,
    }
    if indent is None:
        json_kwargs["separators"] = (",", ":")
    else:
        json_kwargs["indent"] = indent
    return json.dumps(payload, **json_kwargs)


def _project_legacy_v1_wire_shapes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("config_version") != _V1_CONFIG_VERSION:
        return payload

    _strip_classification_metadata(payload)

    structural = payload.get("structural_causal_state")
    if isinstance(structural, dict):
        structural["monetary_pressure"] = {
            "label": "unknown",
            "reason": "not_implemented_v1",
        }
    payload["network_fragility"] = {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    return payload


def _strip_classification_metadata(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("classification_status", None)
        value.pop("classification_reason", None)
        for nested in value.values():
            _strip_classification_metadata(nested)
    elif isinstance(value, list):
        for item in value:
            _strip_classification_metadata(item)


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
    credit_funding_state_proxy: CreditFundingOutput | None = None  # v2 §2C proxy (Log #71)
    credit_funding_effective_state: CreditFundingOutput | None = None  # v2 §2C downstream OAS/proxy resolver
    volume_liquidity_state: VolumeLiquidityStateOutput | None = None  # v2 §1E (slice 2.7)
    monetary_pressure_state: MonetaryPressureV2Output | None = None  # v2 §2A (Log #46)
    change_point: ChangePointOutput | None = None  # v2 §4.6 (V2.1)
    cluster: ClusterOutput | None = None  # v2 §6.2 (Slice 7) — diagnostic evidence
    agent_routing: "AgentRouting | None" = None  # v2 §5.1 (slice 5.1)
    strategy_family_constraints: dict[str, StrategyFamilyConstraint] | None = None  # v2 §5.2 (slice 5.2)

    def model_dump_legacy_v1_wire(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compatibility projection for archived V1 wire-shape replay."""
        kwargs.setdefault("exclude_none", True)
        payload = super().model_dump(*args, **kwargs)
        return _project_legacy_v1_wire_shapes(payload)

    # V1 wire contract: omit any None-valued conditional fields in nested models.
    # Existing callers still receive the archived compatibility projection.
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump_legacy_v1_wire(*args, **kwargs)

    def model_dump_json_legacy_v1_wire(self, *args: Any, **kwargs: Any) -> str:
        indent = kwargs.pop("indent", None)
        ensure_ascii = kwargs.pop("ensure_ascii", False)
        kwargs.setdefault("mode", "json")
        return _dump_json_payload(
            self.model_dump_legacy_v1_wire(*args, **kwargs),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        return self.model_dump_json_legacy_v1_wire(*args, **kwargs)


class RegimeTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str
    config_version: str
    market: str
    start_date: date
    end_date: date
    trading_calendar: str
    outputs: list[RegimeOutput]

    def model_dump_legacy_v1_wire(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compatibility projection for archived V1 wire-shape replay."""
        kwargs.setdefault("exclude_none", True)
        payload = super().model_dump(*args, **kwargs)
        payload["outputs"] = [
            _project_legacy_v1_wire_shapes(output)
            for output in payload.get("outputs", [])
        ]
        return payload

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump_legacy_v1_wire(*args, **kwargs)

    def model_dump_json_legacy_v1_wire(self, *args: Any, **kwargs: Any) -> str:
        indent = kwargs.pop("indent", None)
        ensure_ascii = kwargs.pop("ensure_ascii", False)
        kwargs.setdefault("mode", "json")
        return _dump_json_payload(
            self.model_dump_legacy_v1_wire(*args, **kwargs),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        return self.model_dump_json_legacy_v1_wire(*args, **kwargs)
