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
        close_by_date={day: float(context.spy_ohlcv["close"].loc[pd.Timestamp(day)]) for day in sessions},
        sma_50_by_date={
            day: None if pd.isna(feature_store.sma_50.loc[pd.Timestamp(day)]) else float(feature_store.sma_50.loc[pd.Timestamp(day)])
            for day in sessions
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
    stable_changed_by_date: dict[date, bool] = {}
    days_since_axis_switch_by_date: dict[date, int | None] = {}
    prior_bear_by_date: dict[date, bool] = {}
    last_switch_index: int | None = None
    stable_keys = (
        trend_direction_stable_by_date,
        trend_character_stable_by_date,
        volatility_stable_by_date,
        breadth_stable_by_date,
    )
    for idx, day in enumerate(sessions):
        stable_changed_today = False
        if idx > 0:
            prev_day = sessions[idx - 1]
            stable_changed_today = any(key[day] != key[prev_day] for key in stable_keys)
            if stable_changed_today:
                last_switch_index = idx
        days_since_axis_switch = None
        if last_switch_index is not None and last_switch_index >= max(0, idx - 59):
            days_since_axis_switch = idx - last_switch_index
        stable_changed_by_date[day] = stable_changed_today
        days_since_axis_switch_by_date[day] = days_since_axis_switch
        history_start = max(0, idx - 59)
        prior_bear_by_date[day] = any(
            trend_direction_stable_by_date[sessions[hidx]] == "bear"
            for hidx in range(history_start, idx + 1)
        )
    return TransitionRiskHistory(
        stable_changed_by_date=stable_changed_by_date,
        days_since_axis_switch_by_date=days_since_axis_switch_by_date,
        prior_bear_by_date=prior_bear_by_date,
    )
