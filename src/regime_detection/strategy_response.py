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

    position_size_multiplier = 1.0
    allow_trend_following = True
    allow_mean_reversion = True
    leverage_allowed = True
    allow_buy_dip = True
    allow_breakout = True
    allow_shorts = True
    require_confirmation_for_new_longs = False
    require_confirmation_for_shorts = False
    log_for_review = False
    hard_max_loss_required: bool | None = None
    block_weak_signals: bool | None = None
    prefer_cash_or_hedges: bool | None = None
    take_profit_faster: bool | None = None
    allow_leverage_expansion: bool | None = None
    require_breadth_confirmation: bool | None = None
    reason: str | None = None
    modifiers: list[str] = []

    if (
        trend_direction_active == "bull"
        and trend_character_active in {"trending", "mild_trend", "transition"}
        and volatility_state_active in {"low_vol", "normal_vol"}
        and breadth_state_active == "healthy_breadth"
    ):
        position_size_multiplier = 1.0
        allow_trend_following = True
        allow_buy_dip = True
        allow_leverage_expansion = True
        modifiers.append("bull_healthy_low_vol")

    if transition_risk_label == "recovery_attempt":
        position_size_multiplier = 0.5
        allow_trend_following = True
        allow_buy_dip = True
        require_breadth_confirmation = True
        allow_leverage_expansion = False
        modifiers.append("recovery_attempt")

    if trend_character_active in {"chop", "volatile_chop"} and volatility_state_active != "crisis_vol":
        allow_trend_following = False
        allow_mean_reversion = True
        position_size_multiplier = 0.75
        take_profit_faster = True
        modifiers.append("sideways_chop")

    if transition_risk_label == "bull_fragile_warning":
        position_size_multiplier = 0.5
        allow_buy_dip = False
        allow_leverage_expansion = False
        require_confirmation_for_new_longs = True
        modifiers.append("bull_fragile")

    if transition_risk_label == "sideways_stress_warning":
        # v2 §4.0 — sideways_stress is a defensive modifier for stressed-but-not-bear
        # regimes (banking-crisis, election-uncertainty, macro-shock per spec §4.0).
        # Sits between bull_fragile (mid-conviction defensive) and bear_stress (full
        # defensive) in the strategy_response precedence order at spec §4.0 line 3621.
        # Pattern: trend=sideways AND vol=high AND breadth in {weak_breadth,
        # divergent_fragile} (spec §4.0 lines 3587-3589).
        position_size_multiplier = 0.5
        allow_breakout = False
        allow_leverage_expansion = False
        take_profit_faster = True
        require_confirmation_for_new_longs = True
        modifiers.append("sideways_stress")

    if transition_risk_label == "bear_stress_warning":
        allow_buy_dip = False
        position_size_multiplier = 0.5
        leverage_allowed = False
        require_confirmation_for_shorts = True
        modifiers.append("bear_stress")

    if transition_risk_label == "crisis_override":
        position_size_multiplier = 0.25
        leverage_allowed = False
        hard_max_loss_required = True
        block_weak_signals = True
        prefer_cash_or_hedges = True
        allow_buy_dip = False
        modifiers.append("crisis")

    return StrategyResponse(
        position_size_multiplier=position_size_multiplier,
        allow_trend_following=allow_trend_following,
        allow_mean_reversion=allow_mean_reversion,
        leverage_allowed=leverage_allowed,
        allow_buy_dip=allow_buy_dip,
        allow_breakout=allow_breakout,
        allow_shorts=allow_shorts,
        require_confirmation_for_new_longs=require_confirmation_for_new_longs,
        require_confirmation_for_shorts=require_confirmation_for_shorts,
        log_for_review=log_for_review,
        modifiers_applied=modifiers,
        hard_max_loss_required=hard_max_loss_required,
        block_weak_signals=block_weak_signals,
        prefer_cash_or_hedges=prefer_cash_or_hedges,
        take_profit_faster=take_profit_faster,
        allow_leverage_expansion=allow_leverage_expansion,
        require_breadth_confirmation=require_breadth_confirmation,
        reason=reason,
    )
