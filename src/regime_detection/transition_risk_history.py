from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import pandas as pd

from regime_detection.models import TransitionRiskOutput


@dataclass(frozen=True)
class TransitionRiskHistory:
    stable_changed_by_date: dict[date, bool]
    days_since_axis_switch_by_date: dict[date, int | None]
    axis_switch_count_by_date: dict[date, int]
    recent_axis_switch_count_by_date: dict[date, int]
    prior_bear_by_date: dict[date, bool]


def apply_transition_state_debounce(
    *,
    sessions: list[date],
    raw_outputs: dict[date, TransitionRiskOutput],
    state_confirmation_days: dict[str, int],
    initial_active_state: str | None = None,
) -> dict[date, TransitionRiskOutput]:
    outputs: dict[date, TransitionRiskOutput] = {}
    if initial_active_state is None:
        raise RuntimeError(
            "transition_score.initial_active_state is required for restart-safe "
            "transition state debounce"
        )
    if (
        initial_active_state is not None
        and initial_active_state not in state_confirmation_days
    ):
        raise ValueError(
            f"initial_active_state {initial_active_state!r} not present in "
            f"state_confirmation_days {sorted(state_confirmation_days)}"
        )
    active_state: str | None = initial_active_state
    pending_state: str | None = None
    pending_count = 0

    for day in sessions:
        raw = raw_outputs[day]
        required = state_confirmation_days.get(raw.state)
        if required is None:
            raise ValueError(
                f"transition_score.state_confirmation_days missing state {raw.state!r}"
            )
        if required < 1:
            raise ValueError(
                "transition_score.state_confirmation_days values must be >= 1"
            )

        if active_state is None or raw.state == active_state:
            active_state = raw.state
            pending_state = None
            pending_count = 0
            outputs[day] = raw
            continue

        if raw.state != pending_state:
            pending_state = raw.state
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= required:
            active_state = raw.state
            pending_state = None
            pending_count = 0
            outputs[day] = raw
            continue

        rules = [*raw.triggered_rules, "state_confirmation_pending"]
        outputs[day] = raw.model_copy(
            update={
                "state": active_state,
                "triggered_rules": rules,
                "evidence": raw.evidence.model_copy(update={"triggered_rules": rules}),
            }
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
            "trend_direction": [
                trend_direction_stable_by_date[day] for day in sessions
            ],
            "trend_character": [
                trend_character_stable_by_date[day] for day in sessions
            ],
            "volatility": [volatility_stable_by_date[day] for day in sessions],
            "breadth": [breadth_stable_by_date[day] for day in sessions],
        },
        index=index,
    )

    axis_changed = stable_frame.ne(  # pyright: ignore[reportUnknownMemberType]
        stable_frame.shift(1)
    )
    axis_switch_count = axis_changed.sum(axis=1).astype("int64")
    stable_changed = axis_switch_count.gt(0)
    if not stable_changed.empty:
        stable_changed.iloc[0] = False
        axis_switch_count.iloc[0] = 0

    position = pd.Series(range(len(sessions)), index=index, dtype="int64")
    last_switch_position = pd.Series(
        position.where(stable_changed, -1).cummax().to_numpy(),
        index=index,
        dtype="int64",
    )
    delta = position - last_switch_position
    within_60_sessions = last_switch_position.ge(0) & last_switch_position.ge(
        position - 59
    )
    days_since_axis_switch = delta.where(within_60_sessions)

    # v1 §9.4 recovery_attempt clause: "trend_direction.stable_label was bear
    # at any point in the prior 60 NYSE trading days (excluding as_of_date)".
    # `.shift(1)` drops today from the lookback so the recovery rule only fires
    # when the bear print is in the PAST — preventing recovery_attempt from
    # firing while today's stable_label is still bear (a transition-window
    # edge case during hysteresis lag).
    prior_bear = cast(
        pd.Series,
        stable_frame["trend_direction"]
        .eq("bear")
        .shift(1, fill_value=False)
        .rolling(window=60, min_periods=1)
        .max()
        .astype(bool),
    )

    stable_changed_by_date = {
        cast(date, day): bool(value) for day, value in stable_changed.items()
    }
    axis_switch_count_by_date = {
        cast(date, day): int(value) for day, value in axis_switch_count.items()
    }
    recent_axis_switch_count = (
        axis_switch_count.rolling(window=5, min_periods=1).sum().astype("int64")
    )
    recent_axis_switch_count_by_date = {
        cast(date, day): int(value) for day, value in recent_axis_switch_count.items()
    }
    days_since_axis_switch_by_date = {
        cast(date, day): None if pd.isna(value) else int(value)
        for day, value in days_since_axis_switch.items()
    }
    prior_bear_by_date = {
        cast(date, day): bool(value) for day, value in prior_bear.items()
    }
    return TransitionRiskHistory(
        stable_changed_by_date=stable_changed_by_date,
        days_since_axis_switch_by_date=days_since_axis_switch_by_date,
        axis_switch_count_by_date=axis_switch_count_by_date,
        recent_axis_switch_count_by_date=recent_axis_switch_count_by_date,
        prior_bear_by_date=prior_bear_by_date,
    )


_apply_transition_state_debounce = apply_transition_state_debounce
