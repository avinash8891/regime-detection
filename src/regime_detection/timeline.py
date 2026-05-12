from __future__ import annotations

from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.config import RegimeConfig
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import MarketContext, slice_context_to_recent_sessions
from regime_detection.models import (
    DataQuality,
    MonetaryPressureOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    RegimeTimeline,
    StructuralCausalState,
)
from regime_detection.strategy_response import build_strategy_response
from regime_detection.transition_risk_series import build_transition_risk_series
from regime_detection.versioning import engine_version


ENGINE_MINIMUM_HISTORY = 320


def _v2_classifier_not_yet_implemented_data_quality() -> DataQuality:
    """DataQuality for V2 axes whose classifier hasn't shipped yet.

    Mirrors V1 §2.7 NaN cold-start contract (status=insufficient_history,
    reason=required_feature_is_nan, freshness/completeness null).
    """
    return DataQuality(
        status="insufficient_history",
        freshness_days=None,
        completeness=None,
        reason="required_feature_is_nan",
    )


def _resolve_network_fragility_by_date(
    *,
    bundle_entry: dict[date, NetworkFragilityOutput] | None,
    sessions,
) -> dict[date, NetworkFragilityOutput]:
    """Per-day fragility outputs.

    Prefer the AxisSeriesBundle entry when present (slice 1+ supplies real
    classifications). Fall back to a v2 'unknown' placeholder per session
    when sector ETF data wasn't passed and the bundle entry is None.
    """
    if bundle_entry is not None:
        return bundle_entry
    placeholder_dq = _v2_classifier_not_yet_implemented_data_quality()
    return {
        day: NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"reason": "v2_classifier_not_yet_implemented"},
            data_quality=placeholder_dq,
        )
        for day in sessions
    }


def build_regime_timeline(
    *,
    context: MarketContext,
    lookback_days: int,
    config: RegimeConfig | None = None,
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
    network_fragility_config = (
        config.network_fragility if config is not None else None
    )
    trend_direction_v2_config = (
        config.trend_direction_v2 if config is not None else None
    )
    feature_store = build_feature_store(
        working_context,
        network_fragility_config=network_fragility_config,
        trend_direction_v2_config=trend_direction_v2_config,
    )
    axis_bundle = build_axis_series_bundle(context=working_context, feature_store=feature_store)
    transition_risk = build_transition_risk_series(
        context=working_context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    selected_days = list(working_context.sessions[-lookback_days:])
    trend_direction_outputs = axis_bundle.trend_direction.outputs_by_date
    trend_character_outputs = axis_bundle.trend_character.outputs_by_date
    volatility_outputs = axis_bundle.volatility_state.outputs_by_date
    breadth_outputs = axis_bundle.breadth_state.outputs_by_date
    event_outputs = axis_bundle.event_calendar
    network_fragility_by_date = _resolve_network_fragility_by_date(
        bundle_entry=axis_bundle.network_fragility,
        sessions=working_context.sessions,
    )
    monetary_pressure = MonetaryPressureOutput(
        label="unknown",
        evidence={"reason": "v2_classifier_not_yet_implemented"},
        data_quality=_v2_classifier_not_yet_implemented_data_quality(),
    )

    outputs: list[RegimeOutput] = []
    for day in selected_days:
        trend_direction_output = trend_direction_outputs[day]
        trend_character_output = trend_character_outputs[day]
        volatility_output = volatility_outputs[day]
        breadth_output = breadth_outputs[day]
        event_output = event_outputs[day]
        transition_output = transition_risk[day]
        network_fragility_output = network_fragility_by_date[day]
        outputs.append(
            RegimeOutput(
                engine_version=engine_version(),
                config_version=working_context.config.config_version,
                as_of_date=day,
                market="SPY",
                trend_direction=trend_direction_output,
                trend_character=trend_character_output,
                volatility_state=volatility_output,
                breadth_state=breadth_output,
                structural_causal_state=StructuralCausalState(
                    event_calendar=event_output,
                    monetary_pressure=monetary_pressure,
                ),
                network_fragility=network_fragility_output,
                transition_risk=transition_output,
                strategy_response=build_strategy_response(
                    trend_direction_active=trend_direction_output.active_label,
                    trend_character_active=trend_character_output.active_label,
                    volatility_state_active=volatility_output.active_label,
                    breadth_state_active=breadth_output.active_label,
                    transition_risk_label=transition_output.label,
                    event_calendar_active=event_output.active_label,
                ),
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
