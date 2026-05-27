from __future__ import annotations

from collections.abc import Mapping

from regime_detection.feature_store import FeatureAvailability
from regime_detection.models import (
    AxisCoverage,
    AxisOutput,
    ClassificationCoverageReport,
    RegimeOutput,
    TransitionRiskOutput,
)

_UNSAFE_STATUSES = {
    "data_unavailable",
    "stale_data",
    "insufficient_history",
    "not_wired",
}

_AXIS_FIELDS: tuple[tuple[str, str], ...] = (
    ("trend_direction", "trend_direction"),
    ("trend_character", "trend_character"),
    ("volatility_state", "volatility"),
    ("breadth_state", "breadth"),
    ("network_fragility", "network_fragility"),
    ("volume_liquidity_state", "volume_liquidity_v2"),
    ("credit_funding_state", "credit_funding"),
    ("credit_funding_state_proxy", "credit_funding"),
    ("credit_funding_effective_state", "credit_funding"),
    ("inflation_growth_state", "inflation_growth"),
    ("monetary_pressure_state", "monetary"),
)


def build_classification_coverage(
    output: RegimeOutput,
    *,
    availability: Mapping[str, FeatureAvailability] | None = None,
) -> ClassificationCoverageReport:
    axes: dict[str, AxisCoverage] = {}
    for output_field, feature_name in _AXIS_FIELDS:
        axis_output = getattr(output, output_field)
        feature_availability = (
            None if availability is None else availability.get(feature_name)
        )
        axes[output_field] = _axis_coverage(
            axis=output_field,
            output=axis_output,
            availability=feature_availability,
        )
    axes["transition_risk"] = _transition_risk_coverage(output.transition_risk)
    safe_for_downstream = all(axis.safe_for_downstream for axis in axes.values())
    return ClassificationCoverageReport(
        axes=axes,
        safe_for_downstream=safe_for_downstream,
    )


def _axis_coverage(
    *,
    axis: str,
    output: AxisOutput | None,
    availability: FeatureAvailability | None,
) -> AxisCoverage:
    if output is None:
        safe = availability is not None and availability.policy == "none"
        return AxisCoverage(
            axis=axis,
            status="not_wired",
            label=None,
            reason=availability.reason if availability is not None else "not_emitted",
            safe_for_downstream=safe,
            availability_policy=(
                availability.policy if availability is not None else None
            ),
            required_inputs=(
                availability.required_inputs if availability is not None else ()
            ),
            missing_inputs=(
                availability.missing_inputs if availability is not None else ()
            ),
        )
    status = output.classification_status or "not_wired"
    return AxisCoverage(
        axis=axis,
        status=status,
        label=output.active_label,
        reason=output.classification_reason,
        safe_for_downstream=status not in _UNSAFE_STATUSES,
        availability_policy=availability.policy if availability is not None else None,
        required_inputs=(
            availability.required_inputs if availability is not None else ()
        ),
        missing_inputs=availability.missing_inputs if availability is not None else (),
    )


def _transition_risk_coverage(output: TransitionRiskOutput) -> AxisCoverage:
    status = output.classification_status or "not_wired"
    return AxisCoverage(
        axis="transition_risk",
        status=status,
        label=output.state,
        reason=output.data_quality.reason,
        safe_for_downstream=status not in _UNSAFE_STATUSES,
    )
