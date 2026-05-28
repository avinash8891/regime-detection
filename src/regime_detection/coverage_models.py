from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from regime_detection.model_status import ClassificationStatus

CoverageAxisStatus = ClassificationStatus


class AxisCoverage(BaseModel):
    """Operator-facing coverage for one axis on one classification date."""

    model_config = ConfigDict(extra="forbid")

    axis: str
    status: CoverageAxisStatus
    label: str | None = None
    reason: str | None = None
    safe_for_downstream: bool
    availability_policy: str | None = None
    required_inputs: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()


class ClassificationCoverageReport(BaseModel):
    """Per-date classification coverage and downstream safety summary."""

    model_config = ConfigDict(extra="forbid")

    axes: dict[str, AxisCoverage]
    safe_for_downstream: bool
