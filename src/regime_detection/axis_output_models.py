from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from regime_detection.classification_status import (
    DataQuality,
    derive_classification_status,
)
from regime_detection.evidence_payloads import (
    AxisEvidencePayload,
    CreditFundingEvidencePayload,
    EventCalendarEvidencePayload,
    InflationGrowthEvidencePayload,
    MonetaryPressureEvidencePayload,
    NetworkFragilityEvidencePayload,
    VolumeLiquidityEvidencePayload,
)
from regime_detection.model_status import ClassificationStatus


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
        if self.classification_status in {None, "no_rule_fired"}:
            status, reason = derive_classification_status(
                active_label=self.active_label,
                raw_label=self.raw_label,
                stable_label=self.stable_label,
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

    primary_label: str
    matching_labels: tuple[str, ...]
    evidence: EventCalendarEvidencePayload


class NetworkFragilityOutput(AxisOutput):
    """Layer 3 network fragility classifier output (v2 spec §3).

    The v2 fragility classifier is implemented and wired. `unknown` is reserved
    for data-quality failures or an explicit unpartitioned-rule diagnostic.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["sector_cross_asset_24"] = "sector_cross_asset_24"
    evidence: NetworkFragilityEvidencePayload  # pyright: ignore[reportIncompatibleVariableOverride]


InflationGrowthLabel = Literal[
    "goldilocks",
    "inflation_shock",
    "disinflation",
    "recession_scare",
    "risk_off_mild",
    "recovery_growth",
    "recovery_growth_unconfirmed",
    "reflation",
    "late_cycle_inflation_stress",
    "stagflation_lite",
    "contractionary_disinflation",
    "macro_neutral",
    "earnings_expansion",
    "earnings_contraction",
    "unknown",
]


class InflationGrowthOutput(AxisOutput):
    """v2 §2B inflation/growth axis output.

    Three-tier label triple (raw/stable/active) per the v2 axis pattern.
    ``evidence`` carries the per-day rule inputs and the bias-warning code
    (``commodity_proxy_dbc_substitute``) when applicable. The
    ``earnings_expansion``/``earnings_contraction`` labels consume the weekly
    aggregate forward-EPS revision series when it is wired and naturally
    falsify while that series is absent or in accumulator cold-start.
    """

    model_config = ConfigDict(extra="forbid")

    # fmt: off
    raw_label: InflationGrowthLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    stable_label: InflationGrowthLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    active_label: InflationGrowthLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    evidence: InflationGrowthEvidencePayload  # pyright: ignore[reportIncompatibleVariableOverride]
    # fmt: on


CreditFundingLabel = Literal[
    "credit_calm",
    "credit_recovery",
    "credit_divergence",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


class CreditFundingOutput(AxisOutput):
    """v2 §2C credit/funding state output.

    Three-tier label triple (raw/stable/active) per the v2 axis pattern.
    ``evidence`` carries the per-day scalar rule inputs and the bias-warning
    code (``credit_spread_ice_bofa_oas_fred``).
    """

    model_config = ConfigDict(extra="forbid")

    # fmt: off
    raw_label: CreditFundingLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    stable_label: CreditFundingLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    active_label: CreditFundingLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    evidence: CreditFundingEvidencePayload  # pyright: ignore[reportIncompatibleVariableOverride]
    # fmt: on


MonetaryPressureV2Label = Literal[
    "tightening_pressure",
    "easing_pressure",
    "rate_shock",
    "neutral_monetary",
    "unknown",
]


class MonetaryPressureV2Output(AxisOutput):
    """v2 §2A monetary-pressure axis output (documented implementation decision).

    Three-tier label triple per the v2 axis pattern (raw/stable/active);
    ``evidence`` carries the per-day scalar rule inputs; ``data_quality``
    follows the V1 §2.7 NaN cold-start contract.
    """

    model_config = ConfigDict(extra="forbid")

    # fmt: off
    raw_label: MonetaryPressureV2Label  # pyright: ignore[reportIncompatibleVariableOverride]
    stable_label: MonetaryPressureV2Label  # pyright: ignore[reportIncompatibleVariableOverride]
    active_label: MonetaryPressureV2Label  # pyright: ignore[reportIncompatibleVariableOverride]
    evidence: MonetaryPressureEvidencePayload  # pyright: ignore[reportIncompatibleVariableOverride]
    # fmt: on


class VolumeLiquidityOutput(BaseModel):
    """Volume / liquidity internals output (v2 spec §1E)."""

    model_config = ConfigDict(extra="forbid")

    label: str
    evidence: VolumeLiquidityEvidencePayload
    data_quality: DataQuality
    classification_status: ClassificationStatus | None = None
    classification_reason: str | None = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)

    @property
    def reporting_label(self) -> str:
        if self.classification_status == "classified":
            return self.label
        return self.classification_status or "not_wired"

    @model_validator(mode="after")
    def _populate_classification_metadata(self) -> "VolumeLiquidityOutput":
        if self.classification_status in {None, "no_rule_fired"}:
            status, reason = derive_classification_status(
                active_label=self.label,
                raw_label=self.label,
                stable_label=self.label,
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
    """v2 §1E volume/liquidity state output.

    Carries the three-tier label triple (raw/stable/active) the v2
    axes use, plus per-day evidence and a data-quality record. The
    ``mode`` literal pins the compute path: a z-score over SPY's daily
    share volume (`volume_zscore_v1`). When the feature seam is None
    (no volume column) the timeline emits an unknown-gate output via
    the engine wiring rather than instantiating this output class.
    """

    model_config = ConfigDict(extra="forbid")

    # fmt: off
    raw_label: VolumeLiquidityLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    stable_label: VolumeLiquidityLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    active_label: VolumeLiquidityLabel  # pyright: ignore[reportIncompatibleVariableOverride]
    evidence: VolumeLiquidityEvidencePayload  # pyright: ignore[reportIncompatibleVariableOverride]
    # fmt: on
    mode: Literal["volume_zscore_v1"] = "volume_zscore_v1"


class ClusterOutput(BaseModel):
    """v2 §6.2 clustering output. Diagnostic evidence; per-day
    cluster assignment + Mahalanobis distance to the assigned-cluster
    centroid. ``mapped_label`` is populated when an operator-curated
    ``cluster_label_map.yaml`` is loaded (v2 §6.2 mapping artifact at
    spec line 4233 + V2 §10 at spec line 4359);
    None when the map is absent or still in candidate state.
    """

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    distance_to_centroid: float
    model_version: str
    mapped_label: str | None = None
    mapping_status: Literal[
        "mapped",
        "map_absent",
        "map_invalid",
        "model_version_mismatch",
        "map_required_missing",
    ]
    mapping_reason: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class HmmOutput(BaseModel):
    """v2 §6.1 HMM evidence output.

    Surfaces the Gaussian HMM state assignment for downstream consumers.
    ``mapped_label`` is populated when an operator-curated
    ``hmm_state_label_map.yaml`` is loaded (§6.1 + §10); None otherwise.
    ``state_persistence_days`` counts consecutive sessions the top state
    has remained unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    top_state: int
    top_state_prob: float
    n_states: int
    state_persistence_days: int | None = None
    model_version: str
    mapped_label: str | None = None
    mapping_status: Literal[
        "mapped",
        "map_absent",
        "map_invalid",
        "model_version_mismatch",
        "map_required_missing",
    ]
    mapping_reason: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


class ChangePointOutput(BaseModel):
    """v2 §4.6 + §6.3 BOCPD change-point detection output (evidence-only).

    score: 5-session rolling max of recent short-run BOCPD posterior mass.
    days_since_last_break: int sessions since last posterior mass >= break_threshold.
        None when no break has occurred in the trailing BOCPD window
        (cold-start) — omitted from the JSON wire via exclude_none.
    method: pinned to ``"BOCPD"`` (Adams-MacKay 2007).
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
