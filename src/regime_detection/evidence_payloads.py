from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

from regime_detection.model_status import ClassificationStatus


class EvidencePayload(RootModel[dict[str, Any]]):
    """Dict-compatible named payload for unversioned regime evidence."""

    def get(self, key: str, default: Any = None) -> Any:
        return self.root.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.root[key]

    def __contains__(self, key: object) -> bool:
        return key in self.root

    # fmt: off
    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # fmt: on
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


class EventCalendarEvidencePayload(EvidencePayload):
    """Dict-compatible payload for event-calendar rule evidence."""


class TypedEvidencePayload(BaseModel):
    """Dict-compatible base for typed V2 axis evidence payloads.

    Typed payloads keep the historical mapping ergonomics used by reports while
    forbidding undeclared evidence keys. Add fields here axis-by-axis only after
    the dependency and absence contracts for that axis are declared.
    """

    model_config = ConfigDict(extra="forbid")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.model_dump()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.model_dump()

    # fmt: off
    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # fmt: on
        return iter(self.model_dump())

    def __len__(self) -> int:
        return len(self.model_dump())

    def items(self) -> Any:
        return self.model_dump().items()

    def keys(self) -> Any:
        return self.model_dump().keys()

    def values(self) -> Any:
        return self.model_dump().values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypedEvidencePayload):
            return self.model_dump() == other.model_dump()
        if isinstance(other, dict):
            return self.model_dump() == other
        return NotImplemented


class AxisEvidencePayload(TypedEvidencePayload):
    """Typed payload for core axis rule evidence.

    Core V1 axes are no longer arbitrary evidence bags. This shape declares the
    payload fields that may cross the model/reporting boundary: rule evidence,
    hysteresis metadata, data-quality freeze metadata, breadth provenance, and
    HMM enrichment when the V2 evidence seam is lit.
    """

    rule_evidence: dict[str, Any] | None = None
    risk_rank: dict[str, int] | None = None
    deescalation_days: int | None = None
    reason: str | None = None
    data_quality_freeze: bool | None = None
    data_quality_forced_unknown: bool | None = None
    source: str | None = None
    proxy: str | None = None
    row_provenance_mode: str | None = None
    active_label_source: str | None = None
    hmm_top_state: int | None = None
    hmm_top_state_prob: float | None = None


class NetworkFragilityEvidencePayload(TypedEvidencePayload):
    """Typed evidence payload for network-fragility decisions.

    Cross-axis inputs are labels only; upstream evidence and data_quality do not
    cross this edge unless `AXIS_DEPENDENCY_CONTRACTS` is changed first.
    """

    rule_evidence: dict[str, Any] | None = None
    reason: str | None = None
    breadth_active_label: str | None = None
    volatility_active_label: str | None = None
    credit_funding_active_label: str | None = None
    data_quality_freeze: bool | None = None


class InflationGrowthEvidencePayload(TypedEvidencePayload):
    """Typed evidence payload for inflation/growth decisions.

    `credit_funding_active_label` is the effective downstream label, not the raw
    OAS/proxy evidence payload.
    """

    rule_evidence: dict[str, Any] | None = None
    reason: str | None = None
    goldilocks_limb_evidence: dict[str, Any] | None = None
    credit_funding_active_label: str | None = None
    bias_warning_code: str | None = None
    data_quality_freeze: bool | None = None


class MonetaryPressureEvidencePayload(TypedEvidencePayload):
    """Typed evidence payload for monetary-pressure decisions."""

    rule_evidence: dict[str, Any] | None = None
    reason: str | None = None
    central_bank_text_evidence: dict[str, Any] | None = None
    data_quality_freeze: bool | None = None


class VolumeLiquidityEvidencePayload(TypedEvidencePayload):
    """Typed evidence payload for volume/liquidity decisions."""

    rule_evidence: dict[str, Any] | None = None
    reason: str | None = None
    rule_path: str | None = None
    rule_reason: str | None = None
    data_quality_freeze: bool | None = None


class CreditFundingEvidencePayload(BaseModel):
    """Dict-compatible typed evidence payload for credit/funding decisions.

    Effective credit/funding outputs use this same payload type to record the
    OAS/proxy resolver decision without changing the downstream label contract.
    """

    model_config = ConfigDict(extra="forbid")

    rule_evidence: dict[str, Any] | None = None
    reason: str | None = None
    spread_source: str | None = None
    nfci_daily_carried: float | None = None
    kre_spy_slope_63d: float | None = None
    bias_warning_code: str | None = None
    data_quality_freeze: bool | None = None
    source_used: str | None = None
    agreement_status: str | None = None
    oas_label: str | None = None
    proxy_label: str | None = None
    oas_classification_status: ClassificationStatus | None = None
    proxy_classification_status: ClassificationStatus | None = None
    oas_spread_source: str | None = None
    proxy_spread_source: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump(exclude_none=True).get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.model_dump(exclude_none=True)[key]

    def __contains__(self, key: object) -> bool:
        return key in self.model_dump(exclude_none=True)

    # fmt: off
    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # fmt: on
        return iter(self.model_dump(exclude_none=True))

    def __len__(self) -> int:
        return len(self.model_dump(exclude_none=True))

    def items(self) -> Any:
        return self.model_dump(exclude_none=True).items()

    def keys(self) -> Any:
        return self.model_dump(exclude_none=True).keys()

    def values(self) -> Any:
        return self.model_dump(exclude_none=True).values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CreditFundingEvidencePayload):
            return self.model_dump(exclude_none=True) == other.model_dump(
                exclude_none=True
            )
        if isinstance(other, dict):
            return self.model_dump(exclude_none=True) == other
        return NotImplemented


class TransitionRiskEvidencePayload(BaseModel):
    """Dict-compatible typed evidence payload for transition-risk decisions."""

    model_config = ConfigDict(extra="forbid")

    triggered_rules: list[str]
    stable_changed_today: bool
    days_since_axis_switch: int | None
    axis_switch_count: int
    recent_axis_switch_count: int
    macro_event_labels: list[str] = Field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.model_dump()[key]

    def __contains__(self, key: object) -> bool:
        return key in type(self).model_fields

    # fmt: off
    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # fmt: on
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
