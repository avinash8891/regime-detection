from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BoundaryAbsenceBehavior = Literal["raise", "none", "unknown", "degraded"]


@dataclass(frozen=True)
class BoundaryAbsencePolicy:
    """Declared absence behavior for a boundary that may see missing inputs.

    This registry is not a blanket runtime adapter. It records which existing
    policy is intentional at each boundary so future changes do not collapse
    deterministic errors, optional V2 seams, and data-quality unknowns into one
    behavior.
    """

    name: str
    owner_module: str
    source_symbol: str
    behavior: BoundaryAbsenceBehavior
    rationale: str


BOUNDARY_ABSENCE_POLICIES: tuple[BoundaryAbsencePolicy, ...] = (
    BoundaryAbsencePolicy(
        name="engine.event_calendar",
        owner_module="regime_detection.engine",
        source_symbol="_require_event_calendar",
        behavior="raise",
        rationale=(
            "The regime engine requires normalized event input; absent calendar "
            "data is a caller contract error, not an optional V2 seam."
        ),
    ),
    BoundaryAbsencePolicy(
        name="engine.as_of_trading_day",
        owner_module="regime_detection.engine",
        source_symbol="require_nyse_trading_day",
        behavior="raise",
        rationale=(
            "Classification dates must be NYSE sessions to avoid non-trading-date "
            "rollback and future-data ambiguity."
        ),
    ),
    BoundaryAbsencePolicy(
        name="timeline.lookback_window",
        owner_module="regime_detection.timeline",
        source_symbol="build_regime_timeline",
        behavior="raise",
        rationale=(
            "A requested output window that cannot be represented from available "
            "sessions is an invalid run request."
        ),
    ),
    BoundaryAbsencePolicy(
        name="timeline.network_fragility_absent",
        owner_module="regime_detection.timeline",
        source_symbol="_resolve_network_fragility_by_date",
        behavior="unknown",
        rationale=(
            "Network fragility remains visible on the wire as a V2 unknown "
            "placeholder when its optional sector-data seam is absent."
        ),
    ),
    BoundaryAbsencePolicy(
        name="timeline.monetary_pressure_configured_but_unavailable",
        owner_module="regime_detection.timeline",
        source_symbol="_build_timeline_output_for_day",
        behavior="raise",
        rationale=(
            "If monetary pressure output is configured but unavailable, the "
            "runner is missing required macro inputs and must fail loudly."
        ),
    ),
    BoundaryAbsencePolicy(
        name="feature_store.spy_index",
        owner_module="regime_detection.feature_store",
        source_symbol="_as_datetime_index",
        behavior="raise",
        rationale=(
            "Feature construction requires a DatetimeIndex-backed SPY frame; a "
            "non-date index breaks point-in-time alignment."
        ),
    ),
    BoundaryAbsencePolicy(
        name="feature_store.spy_column",
        owner_module="regime_detection.feature_store",
        source_symbol="_series_column",
        behavior="raise",
        rationale=(
            "Core OHLCV columns are required runtime inputs and cannot be "
            "downgraded to optional seams."
        ),
    ),
    BoundaryAbsencePolicy(
        name="feature_store.required_core_feature",
        owner_module="regime_detection.feature_store",
        source_symbol="_require_feature",
        behavior="raise",
        rationale=(
            "Core feature builders must populate required V1 features before "
            "axis construction starts."
        ),
    ),
    BoundaryAbsencePolicy(
        name="feature_store.optional_v2_seam",
        owner_module="regime_detection.feature_store",
        source_symbol="_FeatureStoreBuildState",
        behavior="none",
        rationale=(
            "Optional V2 feature seams stay None when config or source inputs are "
            "absent so V1 replay and partial V2 runs remain distinguishable."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.network_fragility_seam_absent",
        owner_module="regime_detection.axis_builders.network_fragility",
        source_symbol="build_network_fragility_axis_series",
        behavior="none",
        rationale=(
            "The network fragility axis returns None when its feature/config seam "
            "is unlit; timeline owns the wire-level unknown placeholder."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.volume_liquidity_seam_absent",
        owner_module="regime_detection.axis_builders.volume_liquidity",
        source_symbol="build_volume_liquidity_axis_series",
        behavior="none",
        rationale=(
            "Volume/liquidity is an optional V2 axis and remains absent when "
            "volume inputs or V2 config are absent."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.monetary_pressure_seam_absent",
        owner_module="regime_detection.axis_builders.monetary_pressure",
        source_symbol="build_monetary_pressure_axis_series",
        behavior="none",
        rationale=(
            "Monetary pressure returns None when the V2 feature/config seam is "
            "unlit; configured-but-missing output is handled later by timeline."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.credit_funding_seam_absent",
        owner_module="regime_detection.axis_builders.credit_funding",
        source_symbol="build_credit_funding_axis_series",
        behavior="none",
        rationale=(
            "Credit/funding real and proxy axes are optional V2 seams and return "
            "None when their required feature/config inputs are absent."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.inflation_growth_seam_absent",
        owner_module="regime_detection.axis_builders.inflation_growth",
        source_symbol="build_inflation_growth_axis_series",
        behavior="none",
        rationale=(
            "Inflation/growth remains absent when its V2 feature/config seam is "
            "unlit instead of emitting a synthetic classification."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.data_quality_failure",
        owner_module="regime_detection.axis_builders.per_label",
        source_symbol="quality_forces_unknown",
        behavior="unknown",
        rationale=(
            "When inputs exist but quality fails, axes emit unknown with data "
            "quality evidence rather than disappearing from the output."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.inflation_growth_staleness_gate",
        owner_module="regime_detection.axis_builders.inflation_growth",
        source_symbol="staleness_for_source",
        behavior="unknown",
        rationale=(
            "Stale CPI, PMI, or DGS10 inputs force an unknown inflation/growth "
            "classification with a staleness reason."
        ),
    ),
    BoundaryAbsencePolicy(
        name="axis_builder.credit_funding_staleness_gate",
        owner_module="regime_detection.axis_builders.credit_funding",
        source_symbol="staleness_for_source",
        behavior="unknown",
        rationale=(
            "Stale required credit/funding sources emit unknown rather than "
            "silently using outdated macro or spread data."
        ),
    ),
    BoundaryAbsencePolicy(
        name="transition_risk.v2_score_inputs_required",
        owner_module="regime_detection.transition_risk_series",
        source_symbol="build_transition_risk_series",
        behavior="raise",
        rationale=(
            "When transition_score is configured, missing V2 score inputs mean "
            "the transition risk model cannot be truthfully computed."
        ),
    ),
    BoundaryAbsencePolicy(
        name="transition_score.optional_axis_component",
        owner_module="regime_detection.transition_risk_series",
        source_symbol="_active_label_for_day",
        behavior="none",
        rationale=(
            "Optional axis components are omitted from transition score inputs "
            "when absent or not classified."
        ),
    ),
)


def boundary_absence_policies_by_name() -> dict[str, BoundaryAbsencePolicy]:
    return {policy.name: policy for policy in BOUNDARY_ABSENCE_POLICIES}


__all__ = [
    "BOUNDARY_ABSENCE_POLICIES",
    "BoundaryAbsenceBehavior",
    "BoundaryAbsencePolicy",
    "boundary_absence_policies_by_name",
]
