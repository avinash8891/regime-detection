from __future__ import annotations

from datetime import date

from regime_detection.classification_coverage import build_classification_coverage
from regime_detection.feature_store import FeatureAvailability
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
)


def _dq(status: str = "ok", reason: str | None = None) -> DataQuality:
    return DataQuality(status=status, freshness_days=0, completeness=1.0, reason=reason)


def _axis(label: str, *, status: str = "ok", reason: str | None = None) -> AxisOutput:
    return AxisOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={"rule_evidence": {"source": "test"}},
        data_quality=_dq(status, reason),
    )


def _output() -> RegimeOutput:
    return RegimeOutput(
        engine_version="test",
        config_version="core3-v2.0.0",
        as_of_date=date(2026, 5, 27),
        market="SPY",
        trend_direction=_axis("bull"),
        trend_character=_axis("trending"),
        volatility_state=_axis("normal_vol"),
        breadth_state=BreadthStateOutput(
            raw_label="healthy_breadth",
            stable_label="healthy_breadth",
            active_label="healthy_breadth",
            evidence={"rule_evidence": {"source": "test"}},
            data_quality=_dq(),
            mode="etf_proxy",
        ),
        structural_causal_state=StructuralCausalState(
            event_calendar=EventCalendarOutput(
                primary_label="normal_calendar",
                matching_labels=("normal_calendar",),
                evidence={"selection_method": "precedence"},
            )
        ),
        network_fragility=NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"reason": "sector_etf_closes_missing"},
            data_quality=_dq("insufficient_data", "sector_etf_closes_missing"),
        ),
        transition_risk=TransitionRiskOutput(
            state="stable",
            evidence=TransitionRiskEvidencePayload(
                triggered_rules=[],
                stable_changed_today=False,
                days_since_axis_switch=None,
                axis_switch_count=0,
                recent_axis_switch_count=0,
            ),
            score=0.1,
            score_components={},
            data_quality=_dq(),
        ),
        strategy_response=StrategyResponse(
            position_size_multiplier=1.0,
            allow_trend_following=True,
            allow_mean_reversion=True,
            leverage_allowed=False,
            allow_buy_dip=True,
            allow_breakout=True,
            allow_shorts=False,
            require_confirmation_for_new_longs=False,
            require_confirmation_for_shorts=True,
            log_for_review=False,
            modifiers_applied=[],
        ),
    )


def test_classification_coverage_reports_axis_status_and_safety() -> None:
    availability = {
        "network_fragility": FeatureAvailability(
            feature="network_fragility",
            available=False,
            policy="none",
            reason="missing_required_inputs",
            required_inputs=("sector_etf_closes",),
            missing_inputs=("sector_etf_closes",),
        )
    }

    report = build_classification_coverage(_output(), availability=availability)

    assert report.safe_for_downstream is False
    assert report.axes["trend_direction"].status == "classified"
    assert report.axes["network_fragility"].status == "data_unavailable"
    assert report.axes["network_fragility"].reason == "sector_etf_closes_missing"
    assert report.axes["network_fragility"].availability_policy == "none"
    assert report.axes["network_fragility"].missing_inputs == ("sector_etf_closes",)
