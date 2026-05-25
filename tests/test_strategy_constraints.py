from __future__ import annotations

from regime_detection.models import (
    AgentRouting,
    StrategyFamilyConstraint,
    StrategyResponse,
)
from regime_detection.strategy_constraints import resolve_effective_strategy_constraints


def _permissive_response() -> StrategyResponse:
    return StrategyResponse(
        position_size_multiplier=1.0,
        allow_trend_following=True,
        allow_mean_reversion=True,
        leverage_allowed=True,
        allow_buy_dip=True,
        allow_breakout=True,
        allow_shorts=True,
        require_confirmation_for_new_longs=False,
        require_confirmation_for_shorts=False,
        log_for_review=False,
        modifiers_applied=[],
    )


def test_effective_constraints_block_breakout_when_family_constraint_blocks_it() -> None:
    resolved = resolve_effective_strategy_constraints(
        strategy_response=_permissive_response(),
        agent_routing=AgentRouting(
            active_cohort="default_neutral",
            fallback_cohort="default_neutral",
            blocked_strategy_modes=[],
        ),
        strategy_family_constraints={
            "breakout": StrategyFamilyConstraint(
                allowed=False,
                reason="false_breakout_rate_high_in_chop",
            )
        },
    )

    breakout = resolved["breakout"]
    assert breakout.family == "breakout"
    assert breakout.allowed is False
    assert breakout.sources == [
        "strategy_response",
        "strategy_family_constraints",
    ]
    assert breakout.blocking_reasons == [
        "strategy_family_constraints:false_breakout_rate_high_in_chop",
    ]
    assert breakout.position_size_multiplier == 1.0
    assert breakout.leverage_allowed is True


def test_effective_constraints_block_agent_routed_modes_even_without_family_constraint() -> None:
    resolved = resolve_effective_strategy_constraints(
        strategy_response=_permissive_response(),
        agent_routing=AgentRouting(
            active_cohort="crisis_specialist",
            fallback_cohort="default_neutral",
            blocked_strategy_modes=["leveraged_long"],
        ),
        strategy_family_constraints={},
    )

    leveraged = resolved["leveraged_long"]
    assert leveraged.family == "leveraged_long"
    assert leveraged.allowed is False
    assert leveraged.sources == ["strategy_response", "agent_routing"]
    assert leveraged.blocking_reasons == ["agent_routing:crisis_specialist"]
    assert leveraged.leverage_allowed is True


def test_effective_constraints_apply_strategy_response_blocks() -> None:
    response = _permissive_response().model_copy(
        update={
            "allow_breakout": False,
            "allow_trend_following": False,
            "reason": "sideways_stress",
        }
    )

    resolved = resolve_effective_strategy_constraints(
        strategy_response=response,
        agent_routing=None,
        strategy_family_constraints=None,
    )

    assert resolved["breakout"].allowed is False
    assert resolved["breakout"].blocking_reasons == [
        "strategy_response:sideways_stress",
    ]
    assert resolved["trend_following"].allowed is False
    assert resolved["trend_following"].blocking_reasons == [
        "strategy_response:sideways_stress",
    ]
    assert resolved["mean_reversion"].allowed is True
