from __future__ import annotations

from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import MarketContext, slice_context_to_recent_sessions
from regime_detection.models import LabelReasonOutput, RegimeOutput, RegimeTimeline, StructuralCausalState
from regime_detection.strategy_response import build_strategy_response
from regime_detection.transition_risk_series import build_transition_risk_series
from regime_detection.versioning import engine_version


ENGINE_MINIMUM_HISTORY = 320


def build_regime_timeline(
    *,
    context: MarketContext,
    lookback_days: int,
) -> RegimeTimeline:
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0. Got: {lookback_days}")
    if len(context.sessions) < lookback_days:
        raise ValueError(
            "Insufficient NYSE trading-day coverage for requested lookback_days. "
            f"Requested={lookback_days}, available={len(context.sessions)}, "
            f"end_date={context.end_date.isoformat()}."
        )

    required_sessions = min(len(context.sessions), ENGINE_MINIMUM_HISTORY + lookback_days - 1)
    working_context = slice_context_to_recent_sessions(context=context, required_sessions=required_sessions)
    feature_store = build_feature_store(working_context)
    axis_bundle = build_axis_series_bundle(context=working_context, feature_store=feature_store)
    transition_risk = build_transition_risk_series(
        context=working_context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    selected_days = list(working_context.sessions[-lookback_days:])
    outputs: list[RegimeOutput] = []
    for day in selected_days:
        event_output = axis_bundle.event_calendar[day]
        transition_output = transition_risk[day]
        strategy_response = build_strategy_response(
            trend_direction_active=axis_bundle.trend_direction.outputs_by_date[day].active_label,
            trend_character_active=axis_bundle.trend_character.outputs_by_date[day].active_label,
            volatility_state_active=axis_bundle.volatility_state.outputs_by_date[day].active_label,
            breadth_state_active=axis_bundle.breadth_state.outputs_by_date[day].active_label,
            transition_risk_label=transition_output.label,
            event_calendar_active=event_output.active_label,
        )
        structural = StructuralCausalState(
            event_calendar=event_output,
            monetary_pressure=LabelReasonOutput(
                label="unknown",
                reason="not_implemented_v1",
            ),
        )
        outputs.append(
            RegimeOutput(
                engine_version=engine_version(),
                config_version=working_context.config.config_version,
                as_of_date=day,
                market="SPY",
                trend_direction=axis_bundle.trend_direction.outputs_by_date[day],
                trend_character=axis_bundle.trend_character.outputs_by_date[day],
                volatility_state=axis_bundle.volatility_state.outputs_by_date[day],
                breadth_state=axis_bundle.breadth_state.outputs_by_date[day],
                structural_causal_state=structural,
                network_fragility=LabelReasonOutput(
                    label="not_implemented_v1",
                    reason="breadth_state_used_as_v1_fragility_proxy",
                ),
                transition_risk=transition_output,
                strategy_response=strategy_response,
            )
        )

    return RegimeTimeline(
        engine_version=engine_version(),
        config_version=working_context.config.config_version,
        market="SPY",
        start_date=selected_days[0],
        end_date=selected_days[-1],
        trading_calendar=working_context.config.trading_calendar,
        outputs=outputs,
    )
