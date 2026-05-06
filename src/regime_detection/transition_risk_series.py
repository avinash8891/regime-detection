from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from regime_detection.axis_series import AxisSeriesBundle
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import TransitionRiskOutput
from regime_detection.transition_risk import build_transition_risk_output_from_flags


@dataclass(frozen=True)
class TransitionRiskHistory:
    stable_changed_by_date: dict[date, bool]
    days_since_axis_switch_by_date: dict[date, int | None]
    prior_bear_by_date: dict[date, bool]


def build_transition_risk_series(
    *,
    context: MarketContext,
    feature_store: FeatureStore,
    axis_bundle: AxisSeriesBundle,
) -> dict[date, TransitionRiskOutput]:
    sessions = list(context.sessions)
    session_index = pd.to_datetime(sessions)
    close_series = context.spy_ohlcv["close"].reindex(session_index)
    sma_50_series = feature_store.sma_50.reindex(session_index)
    history = build_transition_risk_history(
        sessions=sessions,
        trend_direction_stable_by_date=axis_bundle.trend_direction.stable_labels_by_date,
        trend_character_stable_by_date=axis_bundle.trend_character.stable_labels_by_date,
        volatility_stable_by_date=axis_bundle.volatility_state.stable_labels_by_date,
        breadth_stable_by_date=axis_bundle.breadth_state.stable_labels_by_date,
    )
    return build_transition_risk_outputs_by_date(
        sessions=sessions,
        trend_direction_active_by_date=axis_bundle.trend_direction.active_labels_by_date,
        trend_character_active_by_date=axis_bundle.trend_character.active_labels_by_date,
        volatility_state_active_by_date=axis_bundle.volatility_state.active_labels_by_date,
        breadth_state_active_by_date=axis_bundle.breadth_state.active_labels_by_date,
        close_by_date={day: float(value) for day, value in zip(sessions, close_series.to_numpy(), strict=True)},
        sma_50_by_date={
            day: None if pd.isna(value) else float(value)
            for day, value in zip(sessions, sma_50_series.to_numpy(), strict=True)
        },
        history=history,
    )


def build_transition_risk_outputs_by_date(
    *,
    sessions: list[date],
    trend_direction_active_by_date: dict[date, str],
    trend_character_active_by_date: dict[date, str],
    volatility_state_active_by_date: dict[date, str],
    breadth_state_active_by_date: dict[date, str],
    close_by_date: dict[date, float | None],
    sma_50_by_date: dict[date, float | None],
    history: TransitionRiskHistory,
) -> dict[date, TransitionRiskOutput]:
    index = pd.Index(sessions)
    trend_direction_active = pd.Series([trend_direction_active_by_date[day] for day in sessions], index=index)
    trend_character_active = pd.Series([trend_character_active_by_date[day] for day in sessions], index=index)
    volatility_state_active = pd.Series([volatility_state_active_by_date[day] for day in sessions], index=index)
    breadth_state_active = pd.Series([breadth_state_active_by_date[day] for day in sessions], index=index)
    close = pd.Series([close_by_date[day] for day in sessions], index=index, dtype="float64")
    sma_50 = pd.Series([sma_50_by_date[day] for day in sessions], index=index, dtype="float64")
    prior_bear = pd.Series([history.prior_bear_by_date[day] for day in sessions], index=index, dtype="bool")
    stable_changed = pd.Series([history.stable_changed_by_date[day] for day in sessions], index=index, dtype="bool")
    days_since_axis_switch = pd.Series(
        [history.days_since_axis_switch_by_date[day] for day in sessions],
        index=index,
        dtype="float64",
    )

    crisis_override = volatility_state_active.eq("crisis_vol")
    bear_stress_warning = (
        trend_direction_active.eq("bear")
        & volatility_state_active.isin(["high_vol", "crisis_vol"])
        & breadth_state_active.isin(["weak_breadth", "divergent_fragile", "unknown"])
    )
    bull_fragile_warning = trend_direction_active.eq("bull") & breadth_state_active.eq("divergent_fragile")
    recovery_attempt = trend_character_active.eq("recovery_attempt") | (
        prior_bear
        & close.gt(sma_50)
        & breadth_state_active.isin(["recovery_breadth", "healthy_breadth"])
    )
    post_switch_cooldown = stable_changed & days_since_axis_switch.notna() & days_since_axis_switch.le(5) & ~crisis_override
    any_unknown = (
        trend_direction_active.eq("unknown")
        | trend_character_active.eq("unknown")
        | volatility_state_active.eq("unknown")
        | breadth_state_active.eq("unknown")
    )

    outputs: dict[date, TransitionRiskOutput] = {}
    for day in sessions:
        switch_days = history.days_since_axis_switch_by_date[day]
        outputs[day] = build_transition_risk_output_from_flags(
            crisis_override=bool(crisis_override.loc[day]),
            bear_stress_warning=bool(bear_stress_warning.loc[day]),
            bull_fragile_warning=bool(bull_fragile_warning.loc[day]),
            recovery_attempt=bool(recovery_attempt.loc[day]),
            post_switch_cooldown=bool(post_switch_cooldown.loc[day]),
            any_unknown=bool(any_unknown.loc[day]),
            stable_changed_today=history.stable_changed_by_date[day],
            days_since_axis_switch=switch_days,
        )
    return outputs


def build_transition_risk_history(
    *,
    sessions: list[date],
    trend_direction_stable_by_date: dict[date, str],
    trend_character_stable_by_date: dict[date, str],
    volatility_stable_by_date: dict[date, str],
    breadth_stable_by_date: dict[date, str],
) -> TransitionRiskHistory:
    index = pd.Index(sessions)
    stable_frame = pd.DataFrame(
        {
            "trend_direction": [trend_direction_stable_by_date[day] for day in sessions],
            "trend_character": [trend_character_stable_by_date[day] for day in sessions],
            "volatility": [volatility_stable_by_date[day] for day in sessions],
            "breadth": [breadth_stable_by_date[day] for day in sessions],
        },
        index=index,
    )

    stable_changed = stable_frame.ne(stable_frame.shift(1)).any(axis=1)
    if not stable_changed.empty:
        stable_changed.iloc[0] = False

    position = pd.Series(range(len(sessions)), index=index, dtype="int64")
    last_switch_position = pd.Series(
        position.where(stable_changed, -1).cummax().to_numpy(),
        index=index,
        dtype="int64",
    )
    delta = position - last_switch_position
    within_60_sessions = last_switch_position.ge(0) & last_switch_position.ge(position - 59)
    days_since_axis_switch = delta.where(within_60_sessions)

    prior_bear = (
        stable_frame["trend_direction"]
        .eq("bear")
        .rolling(window=60, min_periods=1)
        .max()
        .astype(bool)
    )

    stable_changed_by_date = {day: bool(value) for day, value in stable_changed.items()}
    days_since_axis_switch_by_date = {
        day: None if pd.isna(value) else int(value)
        for day, value in days_since_axis_switch.items()
    }
    prior_bear_by_date = {day: bool(value) for day, value in prior_bear.items()}
    return TransitionRiskHistory(
        stable_changed_by_date=stable_changed_by_date,
        days_since_axis_switch_by_date=days_since_axis_switch_by_date,
        prior_bear_by_date=prior_bear_by_date,
    )
