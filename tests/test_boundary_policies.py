from __future__ import annotations

from regime_detection.boundary_policies import (
    BOUNDARY_ABSENCE_POLICIES,
    BoundaryAbsencePolicy,
    boundary_absence_policies_by_name,
)


def test_boundary_absence_policies_declare_distinct_current_behaviors() -> None:
    policies = boundary_absence_policies_by_name()

    assert policies["engine.event_calendar"].behavior == "raise"
    assert policies["feature_store.optional_v2_seam"].behavior == "none"
    assert policies["axis_builder.data_quality_failure"].behavior == "unknown"
    assert policies["timeline.network_fragility_absent"].behavior == "unknown"
    assert policies["transition_score.optional_axis_component"].behavior == "none"


def test_boundary_absence_policies_include_source_and_rationale() -> None:
    for policy in BOUNDARY_ABSENCE_POLICIES:
        assert isinstance(policy, BoundaryAbsencePolicy)
        assert policy.name
        assert policy.owner_module.startswith("regime_detection.")
        assert policy.source_symbol
        assert policy.behavior in {"raise", "none", "unknown", "degraded"}
        assert policy.rationale


def test_required_and_optional_policies_are_not_collapsed() -> None:
    policies = boundary_absence_policies_by_name()

    assert (
        policies["timeline.monetary_pressure_configured_but_unavailable"].behavior
        == "raise"
    )
    assert policies["axis_builder.monetary_pressure_seam_absent"].behavior == "none"
    assert (
        policies["axis_builder.inflation_growth_staleness_gate"].behavior == "unknown"
    )


def test_current_boundary_policy_names_are_complete_for_known_policy_sites() -> None:
    assert set(boundary_absence_policies_by_name()) == {
        "engine.event_calendar",
        "engine.as_of_trading_day",
        "timeline.lookback_window",
        "timeline.network_fragility_absent",
        "timeline.monetary_pressure_configured_but_unavailable",
        "feature_store.spy_index",
        "feature_store.spy_column",
        "feature_store.required_core_feature",
        "feature_store.optional_v2_seam",
        "axis_builder.network_fragility_seam_absent",
        "axis_builder.volume_liquidity_seam_absent",
        "axis_builder.monetary_pressure_seam_absent",
        "axis_builder.credit_funding_seam_absent",
        "axis_builder.inflation_growth_seam_absent",
        "axis_builder.data_quality_failure",
        "axis_builder.inflation_growth_staleness_gate",
        "axis_builder.credit_funding_staleness_gate",
        "transition_risk.v2_score_inputs_required",
        "transition_score.optional_axis_component",
    }
