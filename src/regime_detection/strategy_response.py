from __future__ import annotations

from regime_detection.models import StrategyResponse


def build_strategy_response(
    *,
    trend_direction_active: str,
    trend_character_active: str,
    volatility_state_active: str,
    breadth_state_active: str,
    transition_risk_label: str,
    event_calendar_active: str,
) -> StrategyResponse:
    if (
        transition_risk_label == "unknown"
        or "unknown" in {trend_direction_active, trend_character_active, volatility_state_active, breadth_state_active, event_calendar_active}
    ):
        return StrategyResponse(
            position_size_multiplier=0.75,
            allow_trend_following=True,
            allow_mean_reversion=True,
            leverage_allowed=False,
            allow_buy_dip=True,
            allow_breakout=True,
            allow_shorts=True,
            require_confirmation_for_new_longs=True,
            require_confirmation_for_shorts=True,
            log_for_review=True,
            reason="unknown_or_unmapped_regime",
            modifiers_applied=[],
        )

    state: dict[str, object] = {
        "position_size_multiplier": 1.0,
        "allow_trend_following": True,
        "allow_mean_reversion": True,
        "leverage_allowed": True,
        "allow_buy_dip": True,
        "allow_breakout": True,
        "allow_shorts": True,
        "require_confirmation_for_new_longs": False,
        "require_confirmation_for_shorts": False,
        "log_for_review": False,
    }
    modifiers: list[str] = []

    if (
        trend_direction_active == "bull"
        and trend_character_active in {"trending", "transition"}
        and volatility_state_active in {"low_vol", "normal_vol"}
        and breadth_state_active == "healthy_breadth"
    ):
        state.update(
            {
                "position_size_multiplier": 1.0,
                "allow_trend_following": True,
                "allow_buy_dip": True,
                "allow_leverage_expansion": True,
            }
        )
        modifiers.append("bull_healthy_low_vol")

    if transition_risk_label == "recovery_attempt":
        state.update(
            {
                "position_size_multiplier": 0.5,
                "allow_trend_following": True,
                "allow_buy_dip": True,
                "require_breadth_confirmation": True,
                "allow_leverage_expansion": False,
            }
        )
        modifiers.append("recovery_attempt")

    if trend_character_active == "chop" and volatility_state_active != "crisis_vol":
        state.update(
            {
                "allow_trend_following": False,
                "allow_mean_reversion": True,
                "position_size_multiplier": 0.75,
                "take_profit_faster": True,
            }
        )
        modifiers.append("sideways_chop")

    if transition_risk_label == "bull_fragile_warning":
        state.update(
            {
                "position_size_multiplier": 0.5,
                "allow_buy_dip": False,
                "allow_leverage_expansion": False,
                "require_confirmation_for_new_longs": True,
            }
        )
        modifiers.append("bull_fragile")

    if transition_risk_label == "sideways_stress_warning":
        # v2 §4.0 — sideways_stress is a defensive modifier for stressed-but-not-bear
        # regimes (banking-crisis, election-uncertainty, macro-shock). Sits between
        # bull_fragile (mid-conviction defensive) and bear_stress (full defensive).
        # Pattern (Apr 2025 Treasury vol episode, NY Fed Banking System Vulnerability
        # 2025 Update): trend=sideways AND vol=high AND breadth=weak/divergent_fragile.
        state.update(
            {
                "position_size_multiplier": 0.5,
                "allow_breakout": False,
                "allow_leverage_expansion": False,
                "take_profit_faster": True,
                "require_confirmation_for_new_longs": True,
            }
        )
        modifiers.append("sideways_stress")

    if transition_risk_label == "bear_stress_warning":
        state.update(
            {
                "allow_buy_dip": False,
                "position_size_multiplier": 0.5,
                "leverage_allowed": False,
                "require_confirmation_for_shorts": True,
            }
        )
        modifiers.append("bear_stress")

    if transition_risk_label == "crisis_override":
        state.update(
            {
                "position_size_multiplier": 0.25,
                "leverage_allowed": False,
                "hard_max_loss_required": True,
                "block_weak_signals": True,
                "prefer_cash_or_hedges": True,
                "allow_buy_dip": False,
            }
        )
        modifiers.append("crisis")

    return StrategyResponse(modifiers_applied=modifiers, **state)
