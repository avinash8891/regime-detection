from __future__ import annotations

from datetime import date

import pytest

from regime_detection.transition_risk import (
    build_transition_risk_output_from_flags,
    classify_transition_risk,
)


_AS_OF = date(2024, 1, 2)


def _classify(**overrides):
    kwargs = {
        "as_of_date": _AS_OF,
        "trend_direction_active": "bull",
        "prior_bear_in_last_60_sessions": False,
        "trend_character_active": "trending",
        "volatility_state_active": "normal_vol",
        "breadth_state_active": "healthy_breadth",
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "close": 100.0,
        "sma_50": 95.0,
    }
    kwargs.update(overrides)
    return classify_transition_risk(**kwargs)


@pytest.mark.parametrize(
    ("flags", "expected_label", "expected_warnings"),
    [
        (
            {
                "crisis_override": True,
                "bear_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "crisis_override",
            [
                "crisis_override",
                "bear_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": True,
                "sideways_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "bear_stress_warning",
            [
                "bear_stress_warning",
                "sideways_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
                "post_switch_cooldown",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "sideways_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "sideways_stress_warning",
            [
                "sideways_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
                "post_switch_cooldown",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "bull_fragile_warning": False,
                "recovery_attempt": False,
                "post_switch_cooldown": False,
                "any_unknown": True,
            },
            "unknown",
            [],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "bull_fragile_warning": False,
                "recovery_attempt": False,
                "post_switch_cooldown": False,
                "any_unknown": False,
            },
            "stable",
            [],
        ),
    ],
)
def test_build_transition_risk_output_from_flags_applies_precedence(
    flags,
    expected_label,
    expected_warnings,
) -> None:
    output = build_transition_risk_output_from_flags(
        stable_changed_today=True,
        days_since_axis_switch=0,
        **flags,
    )

    assert output.label == expected_label
    assert output.evidence["warnings_active"] == expected_warnings
    assert output.evidence["stable_changed_today"] is True
    assert output.evidence["days_since_axis_switch"] == 0


def test_classify_transition_risk_bear_stress_warning_predicate() -> None:
    output = _classify(
        trend_direction_active="bear",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
    )

    assert output.label == "bear_stress_warning"
    assert output.evidence["warnings_active"] == ["bear_stress_warning"]


def test_classify_transition_risk_v2_sideways_warning_is_gated() -> None:
    v1_output = _classify(
        trend_direction_active="sideways",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
        allow_v2_warnings=False,
    )
    v2_output = _classify(
        trend_direction_active="sideways",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
        allow_v2_warnings=True,
    )

    assert v1_output.label == "stable"
    assert v2_output.label == "sideways_stress_warning"


def test_classify_transition_risk_recovery_attempt_from_prior_bear_price_and_breadth() -> None:
    output = _classify(
        prior_bear_in_last_60_sessions=True,
        trend_character_active="transition",
        close=101.0,
        sma_50=100.0,
        breadth_state_active="recovery_breadth",
    )

    assert output.label == "recovery_attempt"
    assert output.evidence["warnings_active"] == ["recovery_attempt"]


def test_classify_transition_risk_post_switch_cooldown_is_inclusive_through_day_five() -> None:
    day_five = _classify(days_since_axis_switch=5, stable_changed_today=False)
    day_six = _classify(days_since_axis_switch=6, stable_changed_today=False)

    assert day_five.label == "post_switch_cooldown"
    assert day_six.label == "stable"


def test_classify_transition_risk_crisis_override_suppresses_cooldown_warning() -> None:
    output = _classify(
        volatility_state_active="crisis_vol",
        days_since_axis_switch=0,
        stable_changed_today=True,
    )

    assert output.label == "crisis_override"
    assert output.evidence["warnings_active"] == ["crisis_override"]
