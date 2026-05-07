from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.models import TransitionRiskOutput


def classify_transition_risk(
    *,
    as_of_date: date,
    trend_direction_active: str,
    prior_bear_in_last_60_sessions: bool,
    trend_character_active: str,
    volatility_state_active: str,
    breadth_state_active: str,
    stable_changed_today: bool,
    days_since_axis_switch: int | None,
    close: float | None,
    sma_50: float | None,
) -> TransitionRiskOutput:
    crisis_override = volatility_state_active == "crisis_vol"
    bear_stress_warning = (
        trend_direction_active == "bear"
        and volatility_state_active in {"high_vol", "crisis_vol"}
        and breadth_state_active in {"weak_breadth", "divergent_fragile", "unknown"}
    )
    bull_fragile_warning = trend_direction_active == "bull" and breadth_state_active == "divergent_fragile"
    recovery_attempt = trend_character_active == "recovery_attempt" or (
        prior_bear_in_last_60_sessions
        and close is not None
        and sma_50 is not None
        and not pd.isna(close)
        and not pd.isna(sma_50)
        and close > sma_50
        and breadth_state_active in {"recovery_breadth", "healthy_breadth"}
    )
    post_switch_cooldown = bool(stable_changed_today and days_since_axis_switch is not None and days_since_axis_switch <= 5)
    any_unknown = any(
        label == "unknown"
        for label in [
            trend_direction_active,
            trend_character_active,
            volatility_state_active,
            breadth_state_active,
        ]
    )
    return build_transition_risk_output_from_flags(
        crisis_override=crisis_override,
        bear_stress_warning=bear_stress_warning,
        bull_fragile_warning=bull_fragile_warning,
        recovery_attempt=recovery_attempt,
        post_switch_cooldown=post_switch_cooldown,
        any_unknown=any_unknown,
        stable_changed_today=stable_changed_today,
        days_since_axis_switch=days_since_axis_switch,
    )


def build_transition_risk_output_from_flags(
    *,
    crisis_override: bool,
    bear_stress_warning: bool,
    bull_fragile_warning: bool,
    recovery_attempt: bool,
    post_switch_cooldown: bool,
    any_unknown: bool,
    stable_changed_today: bool,
    days_since_axis_switch: int | None,
) -> TransitionRiskOutput:
    warnings_active: list[str] = []
    if crisis_override:
        warnings_active.append("crisis_override")
    if bear_stress_warning:
        warnings_active.append("bear_stress_warning")
    if bull_fragile_warning:
        warnings_active.append("bull_fragile_warning")
    if recovery_attempt:
        warnings_active.append("recovery_attempt")
    if post_switch_cooldown and not crisis_override:
        warnings_active.append("post_switch_cooldown")

    if crisis_override:
        label = "crisis_override"
    elif bear_stress_warning:
        label = "bear_stress_warning"
    elif bull_fragile_warning:
        label = "bull_fragile_warning"
    elif recovery_attempt:
        label = "recovery_attempt"
    elif post_switch_cooldown and not crisis_override:
        label = "post_switch_cooldown"
    elif any_unknown:
        label = "unknown"
    else:
        label = "stable"

    return TransitionRiskOutput(
        label=label,
        evidence={
            "warnings_active": warnings_active,
            "stable_changed_today": stable_changed_today,
            "days_since_axis_switch": days_since_axis_switch,
        },
    )
