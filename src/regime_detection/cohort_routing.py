"""v2 §5.1 Agent Cohort Routing precedence walker.

Pins docs/regime_engine_v2_spec.md §5.1 per implementation decision. The walker
traverses ``COHORTS`` top-to-bottom and returns the first
matching cohort's routing decision. ``default_neutral`` is the universal
fallback and always matches when no specialist does.

A predicate referencing an axis whose active label is ``None`` (e.g. an
axis whose feature inputs are absent at this session) returns ``False``
silently — those specialists stay dormant until their inputs land.
"""
from __future__ import annotations

from regime_detection.config import (
    CohortRoutingConfig,
    CohortRoutingRule,
    CohortRoutingRulePredicate,
)
from regime_detection.models import AgentRouting


COHORTS: tuple[str, ...] = (
    "crisis_specialist",
    "euphoria_specialist",
    "bear_stress_specialist",
    "tightening_specialist",
    "easing_specialist",
    "recovery_specialist",
    "chop_mean_reversion_specialist",
    "bull_low_vol_specialist",
    "default_neutral",
)

_FALLBACK = "default_neutral"


def evaluate_cohort_routing(
    *,
    trend_direction_active: str,
    trend_character_active: str,
    volatility_state_active: str,
    breadth_state_active: str,
    network_fragility_active: str,
    monetary_pressure_active: str | None,
    config: CohortRoutingConfig,
) -> AgentRouting:
    """Walk §5.1 precedence top-to-bottom; return the first matching cohort.

    ``default_neutral`` always matches as the universal fallback (§5.1).
    The ``fallback_cohort`` field is always ``default_neutral`` regardless
    of which specialist fires.
    """
    inputs: dict[str, str | None] = {
        "trend_direction": trend_direction_active,
        "trend_character": trend_character_active,
        "volatility_state": volatility_state_active,
        "breadth_state": breadth_state_active,
        "network_fragility": network_fragility_active,
        "monetary_pressure": monetary_pressure_active,
    }
    for cohort in COHORTS:
        if cohort == _FALLBACK:
            break
        rule = config.routing_rules.get(cohort)
        if rule is None:
            continue
        if _rule_matches(rule, inputs):
            return AgentRouting(
                active_cohort=cohort,
                fallback_cohort=_FALLBACK,
                blocked_strategy_modes=list(config.blocked_strategy_modes.get(cohort, ())),
            )
    return AgentRouting(
        active_cohort=_FALLBACK,
        fallback_cohort=_FALLBACK,
        blocked_strategy_modes=list(config.blocked_strategy_modes.get(_FALLBACK, ())),
    )


def _rule_matches(rule: CohortRoutingRule, inputs: dict[str, str | None]) -> bool:
    if not rule.any_of and not rule.all_of:
        return False  # empty rule never matches
    if rule.any_of and not any(_predicate_matches(p, inputs) for p in rule.any_of):
        return False
    if rule.all_of and not all(_predicate_matches(p, inputs) for p in rule.all_of):
        return False
    return True


def _predicate_matches(
    predicate: CohortRoutingRulePredicate, inputs: dict[str, str | None]
) -> bool:
    value = inputs.get(predicate.axis)
    if value is None:
        return False
    return value in predicate.values
