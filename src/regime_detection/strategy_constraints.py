from __future__ import annotations

from regime_detection.models import (
    AgentRouting,
    EffectiveStrategyConstraint,
    StrategyFamilyConstraint,
    StrategyResponse,
)
from regime_detection.strategy_family_constraints import STRATEGY_FAMILIES


_STRATEGY_RESPONSE_ALLOW_FIELD: dict[str, str] = {
    "trend_following": "allow_trend_following",
    "mean_reversion": "allow_mean_reversion",
    "breakout": "allow_breakout",
    "buy_dip": "allow_buy_dip",
    "shorts": "allow_shorts",
}

_FAMILY_FIELDS: tuple[str, ...] = (
    "max_lookback_days",
    "max_holding_days",
    "max_position_pct",
    "min_adx",
    "require_breadth_confirmation",
    "require_volume_confirmation",
    "event_window_only",
)


def resolve_effective_strategy_constraints(
    *,
    strategy_response: StrategyResponse,
    agent_routing: AgentRouting | None,
    strategy_family_constraints: dict[str, StrategyFamilyConstraint] | None,
) -> dict[str, EffectiveStrategyConstraint]:
    """Resolve strategy permissions with most-restrictive-wins precedence.

    The engine emits three legacy strategy guidance surfaces. This resolver is
    the canonical consumer contract: a strategy family or mode is allowed only
    when every available source permits it.
    """
    blocked_modes = set(agent_routing.blocked_strategy_modes if agent_routing else ())
    family_constraints = strategy_family_constraints or {}
    families = list(STRATEGY_FAMILIES)
    for name in _STRATEGY_RESPONSE_ALLOW_FIELD:
        if name not in families:
            families.append(name)
    for name in sorted(blocked_modes):
        if name not in families:
            families.append(name)

    return {
        family: _resolve_one(
            family=family,
            strategy_response=strategy_response,
            agent_routing=agent_routing,
            family_constraint=family_constraints.get(family),
            is_agent_blocked=family in blocked_modes,
        )
        for family in families
    }


def _resolve_one(
    *,
    family: str,
    strategy_response: StrategyResponse,
    agent_routing: AgentRouting | None,
    family_constraint: StrategyFamilyConstraint | None,
    is_agent_blocked: bool,
) -> EffectiveStrategyConstraint:
    sources = ["strategy_response"]
    allowed = True
    blocking_reasons: list[str] = []

    allow_field = _STRATEGY_RESPONSE_ALLOW_FIELD.get(family)
    if allow_field is not None and not bool(getattr(strategy_response, allow_field)):
        allowed = False
        reason = strategy_response.reason or allow_field
        blocking_reasons.append(f"strategy_response:{reason}")

    values = {
        "position_size_multiplier": strategy_response.position_size_multiplier,
        "leverage_allowed": strategy_response.leverage_allowed,
        "require_confirmation_for_new_longs": (
            strategy_response.require_confirmation_for_new_longs
        ),
        "require_confirmation_for_shorts": strategy_response.require_confirmation_for_shorts,
    }

    if family_constraint is not None:
        sources.append("strategy_family_constraints")
        if not family_constraint.allowed:
            allowed = False
            reason = family_constraint.reason or "blocked"
            blocking_reasons.append(f"strategy_family_constraints:{reason}")
        for field in _FAMILY_FIELDS:
            values[field] = getattr(family_constraint, field)

    if is_agent_blocked and agent_routing is not None:
        sources.append("agent_routing")
        allowed = False
        blocking_reasons.append(f"agent_routing:{agent_routing.active_cohort}")

    return EffectiveStrategyConstraint(
        family=family,
        allowed=allowed,
        sources=sources,
        blocking_reasons=blocking_reasons,
        **values,
    )
