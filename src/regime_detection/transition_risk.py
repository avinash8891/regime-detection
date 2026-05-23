from __future__ import annotations

from dataclasses import dataclass

from regime_detection.models import (
    DataQuality,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
    TransitionRiskState,
)
from regime_detection.transition_score import ComposedTransitionScore

_SCORE_BAND_TO_STATE: dict[str, TransitionRiskState] = {
    "stable": "stable",
    "weakening": "weakening",
    "transition_warning": "transition_warning",
    "high": "high_transition_risk",
}

# Backwards-compatible default for callers that don't pass an explicit
# driver threshold; production callers thread the value from
# TransitionScoreConfig.overrides.primary_driver_min instead.
_DEFAULT_DRIVER_THRESHOLD = 0.35


@dataclass(frozen=True)
class TransitionRuleFlags:
    crisis: bool
    bear_stress: bool
    fragile_bull: bool
    recovery_attempt: bool
    sideways_stress: bool
    event_transition_watch: bool
    post_switch_cooldown: bool
    insufficient_data: bool
    stable_changed_today: bool
    days_since_axis_switch: int | None
    axis_switch_count: int
    recent_axis_switch_count: int


def compose_transition_risk_output(
    *,
    score: ComposedTransitionScore,
    flags: TransitionRuleFlags,
    primary_driver_min: float = _DEFAULT_DRIVER_THRESHOLD,
) -> TransitionRiskOutput:
    if score.score is None or score.interpretation is None or score.components is None:
        state = _select_transition_state(
            score_state="insufficient_data",
            flags=flags,
        )
        data_quality = DataQuality(
            status="insufficient_history",
            reason="transition_score_inputs_not_ready",
        )
    else:
        state = _select_transition_state(
            score_state=_SCORE_BAND_TO_STATE[score.interpretation],
            flags=flags,
        )
        data_quality = DataQuality(status="ok")

    triggered_rules = _triggered_rules(flags)
    primary_drivers = _primary_drivers(score.components, threshold=primary_driver_min)
    return TransitionRiskOutput(
        state=state,
        score=score.score,
        score_components=score.components,
        primary_drivers=primary_drivers,
        triggered_rules=triggered_rules,
        evidence=TransitionRiskEvidencePayload(
            triggered_rules=triggered_rules,
            stable_changed_today=flags.stable_changed_today,
            days_since_axis_switch=flags.days_since_axis_switch,
            axis_switch_count=flags.axis_switch_count,
            recent_axis_switch_count=flags.recent_axis_switch_count,
        ),
        data_quality=data_quality,
    )


def _select_transition_state(
    *,
    score_state: TransitionRiskState,
    flags: TransitionRuleFlags,
) -> TransitionRiskState:
    # Preserve the old emergency override: crisis_vol must de-risk immediately,
    # even if another axis is unknown or secondary stress evidence is absent.
    if flags.crisis:
        return "crisis"
    if flags.bear_stress:
        return "bear_stress"
    if flags.fragile_bull:
        return "fragile_bull"
    if flags.recovery_attempt:
        return "recovery_attempt"
    # The old V2 sideways-stress warning is retained as a watch condition,
    # not as a separate public final state.
    if flags.sideways_stress:
        return "watch"
    # Concrete watch evidence is still useful when another axis is unknown.
    if flags.event_transition_watch:
        return "watch"
    if flags.post_switch_cooldown and score_state == "stable":
        return "watch"
    if flags.insufficient_data:
        return "insufficient_data"
    return score_state


def _triggered_rules(flags: TransitionRuleFlags) -> list[str]:
    rules: list[str] = []
    if flags.crisis:
        rules.append("crisis")
    if flags.bear_stress:
        rules.append("bear_stress")
    if flags.fragile_bull:
        rules.append("fragile_bull")
    if flags.recovery_attempt:
        rules.append("recovery_attempt")
    if flags.sideways_stress:
        rules.append("sideways_stress")
    if flags.event_transition_watch:
        rules.append("event_transition_watch")
    if flags.post_switch_cooldown:
        rules.append("post_switch_cooldown")
    if flags.insufficient_data:
        rules.append("insufficient_data")
    return rules


def _primary_drivers(
    components: dict[str, float] | None,
    *,
    threshold: float,
) -> list[str]:
    if components is None:
        return []
    ranked = sorted(
        components.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [name for name, value in ranked if value >= threshold][:3]
