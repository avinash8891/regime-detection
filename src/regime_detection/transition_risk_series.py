from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from regime_detection.axis_series import AxisSeriesBundle
from regime_detection.config import TransitionScoreConfig
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import EventCalendarOutput, TransitionRiskOutput
from regime_detection.transition_risk import build_transition_risk_output_from_flags
from regime_detection.transition_score import compose_transition_score_for_session

EventCalendarLabel = Literal[
    "geopolitical_event",
    "election_window",
    "fed_week",
    "global_rate_decision",
    "budget_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "normal",
    "unknown",
]
EVENT_CALENDAR_LABELS: frozenset[EventCalendarLabel] = frozenset(
    (
        "geopolitical_event",
        "election_window",
        "fed_week",
        "global_rate_decision",
        "budget_week",
        "cpi_week",
        "nfp_week",
        "expiry_week",
        "earnings_season",
        "normal_calendar",
        "normal",
        "unknown",
    )
)


@dataclass(frozen=True)
class TransitionRiskHistory:
    stable_changed_by_date: dict[date, bool]
    days_since_axis_switch_by_date: dict[date, int | None]
    prior_bear_by_date: dict[date, bool]


@dataclass(frozen=True)
class TransitionScoreInputs:
    realized_vol_short: float
    realized_vol_long: float
    pct_above_50dma: float
    avg_pairwise_corr_percentile_504d: float
    drawdown_252d: float
    event_calendar_label: EventCalendarLabel
    hmm_top_state_prob_now: float | None = None
    hmm_top_state_prob_5d_ago: float | None = None
    change_point_score: float | None = None

    def __post_init__(self) -> None:
        if self.event_calendar_label not in EVENT_CALENDAR_LABELS:
            raise ValueError(
                f"unknown event_calendar_label: {self.event_calendar_label}"
            )


def build_transition_risk_series(
    *,
    context: MarketContext,
    feature_store: FeatureStore,
    axis_bundle: AxisSeriesBundle,
) -> dict[date, TransitionRiskOutput]:
    sessions = list(context.sessions)
    session_index = pd.to_datetime(sessions)
    close_series = _strict_lookup_by_sessions(
        series=context.spy_ohlcv["close"],
        session_index=session_index,
        series_name="transition-risk close series",
    )
    sma_50_series = _strict_lookup_by_sessions(
        series=feature_store.sma_50,
        session_index=session_index,
        series_name="transition-risk sma_50 series",
    )
    history = build_transition_risk_history(
        sessions=sessions,
        trend_direction_stable_by_date=axis_bundle.trend_direction.stable_labels_by_date,
        trend_character_stable_by_date=axis_bundle.trend_character.stable_labels_by_date,
        volatility_stable_by_date=axis_bundle.volatility_state.stable_labels_by_date,
        breadth_stable_by_date=axis_bundle.breadth_state.stable_labels_by_date,
    )

    # V2 §4 transition-score inputs — only assembled when ALL five upstream
    # feature seams are lit AND the yaml carries the transition_score block.
    # Otherwise we fall back to V1-only TransitionRiskOutput (score=None).
    volatility_v2 = feature_store.volatility_state_v2
    breadth_v2 = feature_store.breadth_state_v2
    network_fragility = feature_store.network_fragility
    trend_v2 = feature_store.trend_direction_v2
    transition_score_config = context.config.transition_score

    transition_score_inputs_by_date: dict[date, TransitionScoreInputs] | None = None
    if (
        volatility_v2 is not None
        and breadth_v2 is not None
        and breadth_v2.pct_above_50dma is not None
        and network_fragility is not None
        and trend_v2 is not None
        and transition_score_config is not None
    ):
        transition_score_inputs_by_date = _build_transition_score_inputs_by_date(
            sessions=sessions,
            realized_vol_short=volatility_v2.realized_vol_short,
            realized_vol_long=volatility_v2.realized_vol_long,
            pct_above_50dma=breadth_v2.pct_above_50dma,
            avg_pairwise_corr_percentile_504d=(
                network_fragility.avg_pairwise_corr_percentile_504d
            ),
            drawdown_252d=trend_v2.drawdown_252d,
            event_calendar=axis_bundle.event_calendar,
            hmm_top_state_prob=(
                feature_store.hmm.top_state_prob
                if feature_store.hmm is not None
                else None
            ),
            change_point_score=(
                feature_store.change_point.score
                if feature_store.change_point is not None
                else None
            ),
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
        transition_score_inputs_by_date=transition_score_inputs_by_date,
        transition_score_config=transition_score_config,
        # v2 §4.0 — sideways_stress_warning is a V2-only named warning
        # extension. Spec line 3145 forbids backporting to V1, so the rule
        # only fires when the active config_version is NOT the V1 frozen
        # tag. V1 byte-identity preserved when this is False.
        v2_warnings_enabled=context.config.config_version != "core3-v1.0.0",
        cooldown_window_days=(
            transition_score_config.cooldown_window_days
            if transition_score_config is not None
            else 5
        ),
    )


def _build_transition_score_inputs_by_date(
    *,
    sessions: list[date],
    realized_vol_short: pd.Series,
    realized_vol_long: pd.Series,
    pct_above_50dma: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    drawdown_252d: pd.Series,
    event_calendar: dict[date, EventCalendarOutput],
    hmm_top_state_prob: pd.Series | None = None,
    change_point_score: pd.Series | None = None,
) -> dict[date, TransitionScoreInputs]:
    """Materialise the per-session v2 §4.2 score inputs for every NYSE session.

    Reindexes each input series ONCE against the session DatetimeIndex
    then iterates over numpy arrays — avoids the per-session ``.loc[ts]``
    lookup pattern that was the bottleneck eliminated in commit 75ebb63
    (data_quality perf refactor). Missing index entries surface as NaN
    through ``.reindex`` and propagate to the
    ``compose_transition_score_for_session`` cold-start guard.
    """
    session_index = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
    rvs = realized_vol_short.reindex(session_index).to_numpy(dtype=float, na_value=float("nan"))
    rvl = realized_vol_long.reindex(session_index).to_numpy(dtype=float, na_value=float("nan"))
    pct50 = pct_above_50dma.reindex(session_index).to_numpy(dtype=float, na_value=float("nan"))
    corr = avg_pairwise_corr_percentile_504d.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    dd252 = drawdown_252d.reindex(session_index).to_numpy(dtype=float, na_value=float("nan"))

    # v2 §6.1 — bulk-reindex both `top_state_prob[t]` and
    # `top_state_prob[t-5]`. The 5-NYSE-session offset is materialized by
    # shifting the SOURCE series before reindexing so each session t maps
    # to the value at session t-5 (or NaN if absent).
    if hmm_top_state_prob is not None:
        hmm_now = hmm_top_state_prob.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
        hmm_5d_ago = hmm_top_state_prob.shift(5).reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
    else:
        hmm_now = [float("nan")] * len(sessions)
        hmm_5d_ago = [float("nan")] * len(sessions)

    # documented implementation decision — change_point.score (5-session rolling max of
    # BOCPD posterior P(run_length=0); already ∈ [0,1] by construction).
    if change_point_score is not None:
        cp = change_point_score.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
    else:
        cp = [float("nan")] * len(sessions)

    out: dict[date, TransitionScoreInputs] = {}
    for i, day in enumerate(sessions):
        out[day] = TransitionScoreInputs(
            realized_vol_short=float(rvs[i]),
            realized_vol_long=float(rvl[i]),
            pct_above_50dma=float(pct50[i]),
            avg_pairwise_corr_percentile_504d=float(corr[i]),
            drawdown_252d=float(dd252[i]),
            event_calendar_label=event_calendar[day].active_label,
            hmm_top_state_prob_now=_optional_float(hmm_now[i]),
            hmm_top_state_prob_5d_ago=_optional_float(hmm_5d_ago[i]),
            change_point_score=_optional_float(cp[i]),
        )
    return out


def _optional_float(value: object) -> float | None:
    parsed = float(value)
    if pd.isna(parsed):
        return None
    return parsed


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
    transition_score_inputs_by_date: dict[date, TransitionScoreInputs] | None = None,
    transition_score_config: TransitionScoreConfig | None = None,
    v2_warnings_enabled: bool = False,
    cooldown_window_days: int = 5,
) -> dict[date, TransitionRiskOutput]:
    index = pd.Index(sessions)
    trend_direction_active = pd.Series([trend_direction_active_by_date[day] for day in sessions], index=index)
    trend_character_active = pd.Series([trend_character_active_by_date[day] for day in sessions], index=index)
    volatility_state_active = pd.Series([volatility_state_active_by_date[day] for day in sessions], index=index)
    breadth_state_active = pd.Series([breadth_state_active_by_date[day] for day in sessions], index=index)
    close = pd.Series([close_by_date[day] for day in sessions], index=index, dtype="float64")
    sma_50 = pd.Series([sma_50_by_date[day] for day in sessions], index=index, dtype="float64")
    prior_bear = pd.Series([history.prior_bear_by_date[day] for day in sessions], index=index, dtype="bool")
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
    # v2 §4.0 named warning extension. Spec line 3145 forbids backporting to
    # V1; the `v2_warnings_enabled` flag is only True when the active
    # config_version is NOT the V1 frozen tag. V1 byte-identity preserved
    # when False (the rule never fires regardless of input).
    if v2_warnings_enabled:
        sideways_stress_warning = (
            trend_direction_active.eq("sideways")
            & volatility_state_active.eq("high_vol")
            & breadth_state_active.isin(["weak_breadth", "divergent_fragile"])
        )
    else:
        sideways_stress_warning = pd.Series(False, index=index, dtype="bool")
    bull_fragile_warning = trend_direction_active.eq("bull") & breadth_state_active.eq("divergent_fragile")
    recovery_attempt = trend_character_active.eq("recovery_attempt") | (
        prior_bear
        & close.gt(sma_50)
        & breadth_state_active.isin(["recovery_breadth", "healthy_breadth"])
    )
    # v1 §9.4: cooldown is a 5-session window after a switch (days 0..5 inclusive),
    # not a single-day flag. `days_since_axis_switch <= 5` already covers the full
    # window because `days_since_axis_switch_by_date` increments daily after a switch.
    # crisis_override still breaks cooldown.
    post_switch_cooldown = days_since_axis_switch.notna() & days_since_axis_switch.le(cooldown_window_days) & ~crisis_override
    any_unknown = (
        trend_direction_active.eq("unknown")
        | trend_character_active.eq("unknown")
        | volatility_state_active.eq("unknown")
        | breadth_state_active.eq("unknown")
    )

    outputs: dict[date, TransitionRiskOutput] = {}
    for day in sessions:
        switch_days = history.days_since_axis_switch_by_date[day]
        output = build_transition_risk_output_from_flags(
            crisis_override=bool(crisis_override.loc[day]),
            bear_stress_warning=bool(bear_stress_warning.loc[day]),
            sideways_stress_warning=bool(sideways_stress_warning.loc[day]),
            bull_fragile_warning=bool(bull_fragile_warning.loc[day]),
            recovery_attempt=bool(recovery_attempt.loc[day]),
            post_switch_cooldown=bool(post_switch_cooldown.loc[day]),
            any_unknown=bool(any_unknown.loc[day]),
            stable_changed_today=history.stable_changed_by_date[day],
            days_since_axis_switch=switch_days,
        )
        if (
            transition_score_inputs_by_date is not None
            and transition_score_config is not None
        ):
            inputs = transition_score_inputs_by_date[day]
            composed = compose_transition_score_for_session(
                realized_vol_short=inputs.realized_vol_short,
                realized_vol_long=inputs.realized_vol_long,
                pct_above_50dma=inputs.pct_above_50dma,
                avg_pairwise_corr_percentile_504d=(
                    inputs.avg_pairwise_corr_percentile_504d
                ),
                drawdown_252d=inputs.drawdown_252d,
                event_calendar_label=inputs.event_calendar_label,
                hmm_top_state_prob_now=inputs.hmm_top_state_prob_now,
                hmm_top_state_prob_5d_ago=inputs.hmm_top_state_prob_5d_ago,
                change_point_score=inputs.change_point_score,
                config=transition_score_config,
            )
            if composed.score is not None:
                output = output.model_copy(
                    update={
                        "score": composed.score,
                        "score_interpretation": composed.interpretation,
                        "score_components": composed.components,
                    }
                )
        outputs[day] = output
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

    # v1 §9.4 recovery_attempt clause: "trend_direction.stable_label was bear
    # at any point in the prior 60 NYSE trading days (excluding as_of_date)".
    # `.shift(1)` drops today from the lookback so the recovery rule only fires
    # when the bear print is in the PAST — preventing recovery_attempt from
    # firing while today's stable_label is still bear (a transition-window
    # edge case during hysteresis lag).
    prior_bear = (
        stable_frame["trend_direction"]
        .eq("bear")
        .shift(1)
        .rolling(window=60, min_periods=1)
        .max()
        .fillna(False)
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


def _strict_lookup_by_sessions(
    *,
    series: pd.Series,
    session_index: pd.DatetimeIndex,
    series_name: str,
) -> pd.Series:
    source = series.copy()
    source.index = pd.to_datetime(source.index)
    try:
        positions = source.index.get_indexer(session_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{series_name} index is not exactly aligned to NYSE sessions used by transition-risk computation."
        ) from exc
    if (positions < 0).any():
        missing_sessions = [session_index[idx].date().isoformat() for idx, pos in enumerate(positions) if pos < 0][:5]
        raise ValueError(
            f"{series_name} index is not exactly aligned to NYSE sessions used by transition-risk computation. "
            f"Missing exact matches for sessions: {missing_sessions}"
        )
    return pd.Series(source.to_numpy()[positions], index=session_index, name=source.name)
