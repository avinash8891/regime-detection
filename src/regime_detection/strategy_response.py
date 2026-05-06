from __future__ import annotations

from dataclasses import dataclass

from regime_detection.models import StrategyResponse


_SCENARIO_PRECEDENCE_HIGH_TO_LOW = [
    "crisis",
    "bear_stress",
    "bull_fragile",
    "sideways_chop",
    "recovery_attempt",
    "bull_healthy_low_vol",
    "default_neutral",
]


@dataclass(frozen=True)
class StrategyInputs:
    trend_direction_active: str
    trend_character_active: str
    volatility_active: str
    breadth_active: str
    transition_risk_label: str


def build_strategy_response(*, inp: StrategyInputs) -> StrategyResponse:
    # Unknown fallback: V1 uses an explicit conservative default if the regime is not fully mapped.
    if (
        inp.transition_risk_label == "unknown"
        or inp.trend_direction_active == "unknown"
        or inp.trend_character_active == "unknown"
        or inp.volatility_active == "unknown"
        or inp.breadth_active == "unknown"
    ):
        return StrategyResponse(
            position_size_multiplier=0.75,
            leverage_allowed=False,
            allow_trend_following=True,
            allow_buy_dip=True,
            allow_mean_reversion=True,
            allow_breakout=True,
            allow_shorts=True,
            require_confirmation_for_new_longs=True,
            require_confirmation_for_shorts=True,
            log_for_review=True,
            reason="unknown_or_unmapped_regime",
            modifiers_applied=[],
        )

    base = StrategyResponse(
        position_size_multiplier=1.0,
        leverage_allowed=True,
        allow_trend_following=True,
        allow_buy_dip=True,
        allow_mean_reversion=True,
        allow_breakout=True,
        allow_shorts=True,
        require_confirmation_for_new_longs=False,
        require_confirmation_for_shorts=False,
        log_for_review=False,
        modifiers_applied=[],
    )

    scenarios = _matched_scenarios(inp)
    # Apply modifiers in increasing priority order (lowest -> highest), layered on base.
    applied: list[str] = []
    for scenario in reversed(_SCENARIO_PRECEDENCE_HIGH_TO_LOW):
        if scenario not in scenarios:
            continue
        _apply_scenario(base, scenario)
        if scenario != "default_neutral":
            applied.append(scenario)
    base.modifiers_applied = applied
    return base


def _matched_scenarios(inp: StrategyInputs) -> set[str]:
    out: set[str] = {"default_neutral"}
    if inp.transition_risk_label == "crisis_override":
        out.add("crisis")
    if inp.transition_risk_label == "bear_stress_warning":
        out.add("bear_stress")
    if inp.transition_risk_label == "bull_fragile_warning":
        out.add("bull_fragile")
    if inp.trend_character_active == "chop" and inp.volatility_active != "crisis_vol":
        out.add("sideways_chop")
    if inp.transition_risk_label == "recovery_attempt":
        out.add("recovery_attempt")
    if (
        inp.trend_direction_active == "bull"
        and inp.trend_character_active in ["trending", "transition"]
        and inp.volatility_active in ["low_vol", "normal_vol"]
        and inp.breadth_active == "healthy_breadth"
    ):
        out.add("bull_healthy_low_vol")
    return out


def _apply_scenario(resp: StrategyResponse, scenario: str) -> None:
    if scenario == "default_neutral":
        return
    if scenario == "crisis":
        resp.position_size_multiplier = 0.25
        resp.leverage_allowed = False
        resp.hard_max_loss_required = True
        resp.block_weak_signals = True
        resp.prefer_cash_or_hedges = True
        resp.allow_buy_dip = False
        return
    if scenario == "bear_stress":
        resp.allow_buy_dip = False
        resp.position_size_multiplier = 0.5
        resp.leverage_allowed = False
        resp.require_confirmation_for_shorts = True
        return
    if scenario == "bull_fragile":
        resp.position_size_multiplier = 0.5
        resp.allow_buy_dip = False
        resp.allow_leverage_expansion = False
        resp.require_confirmation_for_new_longs = True
        return
    if scenario == "sideways_chop":
        resp.allow_trend_following = False
        resp.allow_mean_reversion = True
        resp.position_size_multiplier = 0.75
        resp.take_profit_faster = True
        return
    if scenario == "recovery_attempt":
        resp.position_size_multiplier = 0.5
        resp.allow_trend_following = True
        resp.allow_buy_dip = True
        resp.require_breadth_confirmation = True
        resp.allow_leverage_expansion = False
        return
    if scenario == "bull_healthy_low_vol":
        resp.position_size_multiplier = 1.0
        resp.allow_trend_following = True
        resp.allow_buy_dip = True
        resp.allow_leverage_expansion = True
        return
    raise ValueError(f"Unknown scenario: {scenario}")

