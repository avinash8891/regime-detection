"""v2 §5.1 Agent Cohort Routing precedence walker.

Pins docs/regime_engine_v2_spec.md §5.1 per implementation decision. The §5.1
precedence is ``crisis_specialist > data_outage_specialist > euphoria > … >
bull_low_vol > default_neutral``. ``crisis_specialist`` is evaluated first and
pre-empts everything; the ``data_outage_specialist`` fail-closed cohort (fires
when any core risk axis is ``unknown``) is then evaluated BEFORE the optimistic
specialist walk, so an optimistic cohort can never route aggressive modes during
a partial data outage (F-005). The remaining specialists are walked top-to-bottom;
``default_neutral`` is the universal fallback when no specialist matches.

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
_CRISIS = "crisis_specialist"
_DATA_OUTAGE = "data_outage_specialist"
_UNKNOWN_SENSITIVE_AXES: tuple[str, ...] = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "network_fragility",
)


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

    def _route(cohort: str) -> AgentRouting:
        return AgentRouting(
            active_cohort=cohort,
            fallback_cohort=_FALLBACK,
            blocked_strategy_modes=list(config.blocked_strategy_modes.get(cohort, ())),
        )

    # §5.1 precedence: crisis_specialist > data_outage_specialist > optimistic
    # specialists > default_neutral. crisis pre-empts everything; the data_outage
    # fail-closed cohort (any core risk axis ``unknown``) is then evaluated BEFORE
    # the optimistic walk so no optimistic cohort (e.g. chop_mean_reversion, whose
    # rule omits a trend_direction predicate) can route leveraged_long / short_vol
    # during a partial data outage (F-005).
    crisis_rule = config.routing_rules.get(_CRISIS)
    if crisis_rule is not None and _rule_matches(crisis_rule, inputs):
        return _route(_CRISIS)
    if any(inputs[axis] == "unknown" for axis in _UNKNOWN_SENSITIVE_AXES):
        return _route(_DATA_OUTAGE)
    for cohort in COHORTS:
        if cohort == _FALLBACK:
            break
        if cohort == _CRISIS:
            continue  # already evaluated above (pre-empts data_outage)
        rule = config.routing_rules.get(cohort)
        if rule is None:
            continue
        if _rule_matches(rule, inputs):
            return _route(cohort)
    return _route(_FALLBACK)


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
