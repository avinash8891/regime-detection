from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.models import TransitionRiskEvidencePayload, TransitionRiskOutput


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
    allow_v2_warnings: bool = False,
) -> TransitionRiskOutput:
    crisis_override = volatility_state_active == "crisis_vol"
    bear_stress_warning = (
        trend_direction_active == "bear"
        and volatility_state_active in {"high_vol", "crisis_vol"}
        and breadth_state_active in {"weak_breadth", "divergent_fragile", "unknown"}
    )
    # v2 §4.0 named warning extension. Captures stressed-but-not-bear regimes
    # (banking-crisis, election-uncertainty, macro-shock) that V1 emits as
    # `stable`. Spec line 3145 explicitly forbids backporting to V1, so the
    # rule is gated by `allow_v2_warnings` (off by default — V1 byte-identity
    # preserved).
    sideways_stress_warning = bool(
        allow_v2_warnings
        and trend_direction_active == "sideways"
        and volatility_state_active == "high_vol"
        and breadth_state_active in {"weak_breadth", "divergent_fragile"}
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
    # v1 §9.4 post_switch_cooldown is a 5-session WINDOW after any axis stable_label
    # changes, not a single-day flag. `days_since_axis_switch <= 5` covers the full
    # window (0 on switch day through 5 inclusive). Crisis_override breaks cooldown
    # in the precedence walker below.
    post_switch_cooldown = bool(
        days_since_axis_switch is not None and days_since_axis_switch <= 5
    )
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
        sideways_stress_warning=sideways_stress_warning,
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
    sideways_stress_warning: bool = False,
) -> TransitionRiskOutput:
    """Compose the transition_risk label from per-rule flags.

    Precedence (highest first):
      crisis_override > bear_stress_warning > sideways_stress_warning
      > bull_fragile_warning > recovery_attempt > post_switch_cooldown
      > unknown > stable

    `sideways_stress_warning` (V2 §4.0) is inserted between bear_stress and
    bull_fragile: it is compositional like bull_fragile but defensively
    skewed (stressed-but-not-bear). When `sideways_stress_warning=False`
    (the V1 default), the precedence collapses to the V1 ordering and
    V1 byte-identity is preserved.
    """
    warnings_active: list[str] = []
    if crisis_override:
        warnings_active.append("crisis_override")
    if bear_stress_warning:
        warnings_active.append("bear_stress_warning")
    if sideways_stress_warning:
        warnings_active.append("sideways_stress_warning")
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
    elif sideways_stress_warning:
        label = "sideways_stress_warning"
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
        evidence=TransitionRiskEvidencePayload(
            warnings_active=warnings_active,
            stable_changed_today=stable_changed_today,
            days_since_axis_switch=days_since_axis_switch,
        ),
    )
