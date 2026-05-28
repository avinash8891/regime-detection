from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

from regime_detection import legacy_v1_wire as _legacy_v1_wire
from regime_detection.axis_output_models import (
    AxisOutput,
    BreadthStateOutput,
    ChangePointOutput,
    ClusterOutput,
    CreditFundingOutput,
    HmmOutput,
    InflationGrowthOutput,
    MonetaryPressureV2Output,
    NetworkFragilityOutput,
    VolumeLiquidityStateOutput,
)
from regime_detection.coverage_models import ClassificationCoverageReport
from regime_detection.strategy_models import (
    AgentRouting,
    EffectiveStrategyConstraint,
    StrategyFamilyConstraint,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskOutput,
)

_dump_json_payload = _legacy_v1_wire.dump_json_payload
_project_legacy_v1_wire_shapes = _legacy_v1_wire.project_legacy_v1_wire_shapes


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
    inflation_growth_state: InflationGrowthOutput | None = None  # v2 §2B
    credit_funding_state: CreditFundingOutput | None = None  # v2 §2C
    credit_funding_state_proxy: CreditFundingOutput | None = None  # v2 §2C proxy
    credit_funding_effective_state: CreditFundingOutput | None = (
        None  # v2 §2C downstream OAS/proxy resolver
    )
    volume_liquidity_state: VolumeLiquidityStateOutput | None = None  # v2 §1E
    monetary_pressure_state: MonetaryPressureV2Output | None = None  # v2 §2A
    change_point: ChangePointOutput | None = None  # v2 §4.6
    hmm: HmmOutput | None = None  # v2 §6.1 — evidence
    cluster: ClusterOutput | None = None  # v2 §6.2 — diagnostic evidence
    agent_routing: "AgentRouting | None" = None  # v2 §5.1
    strategy_family_constraints: dict[str, StrategyFamilyConstraint] | None = (
        None  # v2 §5.2
    )
    effective_strategy_constraints: dict[str, EffectiveStrategyConstraint] | None = None
    classification_coverage: ClassificationCoverageReport | None = None

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
