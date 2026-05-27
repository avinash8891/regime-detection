from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from regime_detection.axis_series import AxisSeriesBundle
from regime_detection.config import TransitionScoreConfig, load_default_regime_config
from regime_detection.event_calendar_labels import (
    EVENT_CALENDAR_LABEL_SET,
    EventCalendarLabel,
)
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import EventCalendarOutput, TransitionRiskOutput
from regime_detection.transition_risk import (
    TransitionRuleFlags,
    compose_transition_risk_output,
)
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compose_transition_score_for_session,
)

EVENT_CALENDAR_LABELS = EVENT_CALENDAR_LABEL_SET


@dataclass(frozen=True)
class TransitionRiskHistory:
    stable_changed_by_date: dict[date, bool]
    days_since_axis_switch_by_date: dict[date, int | None]
    axis_switch_count_by_date: dict[date, int]
    recent_axis_switch_count_by_date: dict[date, int]
    prior_bear_by_date: dict[date, bool]


@dataclass(frozen=True)
class TransitionScoreInputs:
    realized_vol_short: float
    realized_vol_long: float
    pct_above_50dma: float
    avg_pairwise_corr_percentile_504d: float
    drawdown_252d: float
    event_calendar_labels: tuple[EventCalendarLabel, ...]
    spy_close: float | None = None
    spy_sma_50: float | None = None
    largest_eigenvalue_share_percentile_504d: float | None = None
    effective_rank_percentile_504d: float | None = None
    absorption_ratio_top3: float | None = None
    credit_funding_label: str | None = None
    volume_liquidity_label: str | None = None
    volume_zscore_20d: float | None = None
    gap_frequency_percentile_252d: float | None = None
    intraday_range_percentile_252d: float | None = None
    hmm_top_state_prob_now: float | None = None
    hmm_top_state_prob_5d_ago: float | None = None
    change_point_score: float | None = None
    cluster_id_now: int | None = None
    cluster_id_5d_ago: int | None = None

    def __post_init__(self) -> None:
        bad = [
            label
            for label in self.event_calendar_labels
            if label not in EVENT_CALENDAR_LABELS
        ]
        if bad:
            raise ValueError(f"unknown event_calendar_labels: {bad}")


def build_transition_risk_series(
    *,
    context: MarketContext,
    feature_store: FeatureStore,
    axis_bundle: AxisSeriesBundle,
    output_sessions: list[date] | None = None,
) -> dict[date, TransitionRiskOutput]:
    sessions = list(context.sessions)
    requested_output_sessions = (
        None if output_sessions is None else list(output_sessions)
    )
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

    volatility_v2 = feature_store.volatility_state_v2
    breadth_v2 = feature_store.breadth_state_v2
    network_fragility = feature_store.network_fragility
    trend_v2 = feature_store.trend_direction_v2
    transition_score_config = context.config.transition_score
    if transition_score_config is None:
        legacy_selection_config = _legacy_transition_score_config()
        transition_score_inputs_by_date = _build_legacy_transition_score_inputs_by_date(
            sessions=sessions,
            event_calendar=axis_bundle.event_calendar,
            close=close_series,
            sma_50=sma_50_series,
        )
        outputs = build_transition_risk_outputs_by_date(
            sessions=sessions,
            trend_direction_active_by_date=axis_bundle.trend_direction.active_labels_by_date,
            trend_character_active_by_date=axis_bundle.trend_character.active_labels_by_date,
            volatility_state_active_by_date=axis_bundle.volatility_state.active_labels_by_date,
            breadth_state_active_by_date=axis_bundle.breadth_state.active_labels_by_date,
            close_by_date={
                day: float(value)
                for day, value in zip(sessions, close_series.to_numpy(), strict=True)
            },
            sma_50_by_date={
                day: None if pd.isna(value) else float(value)
                for day, value in zip(sessions, sma_50_series.to_numpy(), strict=True)
            },
            history=history,
            transition_score_inputs_by_date=transition_score_inputs_by_date,
            transition_score_config=None,
            cooldown_window_days=legacy_selection_config.cooldown_window_days,
            state_confirmation_days=legacy_selection_config.state_confirmation_days,
            initial_active_state=legacy_selection_config.initial_active_state,
        )
        return _filter_transition_risk_outputs(
            outputs=outputs,
            output_sessions=requested_output_sessions,
        )

    missing = []
    if volatility_v2 is None:
        missing.append("feature_store.volatility_state_v2")
    if breadth_v2 is None:
        missing.append("feature_store.breadth_state_v2")
    elif breadth_v2.pct_above_50dma is None:
        missing.append("feature_store.breadth_state_v2.pct_above_50dma")
    if network_fragility is None:
        missing.append("feature_store.network_fragility")
    if trend_v2 is None:
        missing.append("feature_store.trend_direction_v2")
    if missing:
        raise RuntimeError(
            "transition_risk requires score inputs; missing: " + ", ".join(missing)
        )

    transition_score_inputs_by_date = _build_transition_score_inputs_by_date(
        sessions=sessions,
        realized_vol_short=volatility_v2.realized_vol_short,
        realized_vol_long=volatility_v2.realized_vol_long,
        pct_above_50dma=breadth_v2.pct_above_50dma,
        avg_pairwise_corr_percentile_504d=(
            network_fragility.avg_pairwise_corr_percentile_504d
        ),
        largest_eigenvalue_share_percentile_504d=(
            network_fragility.largest_eigenvalue_share_percentile_504d
        ),
        effective_rank_percentile_504d=(
            network_fragility.effective_rank_percentile_504d
        ),
        absorption_ratio_top3=network_fragility.absorption_ratio_top3,
        drawdown_252d=trend_v2.drawdown_252d,
        event_calendar=axis_bundle.event_calendar,
        close=close_series,
        sma_50=sma_50_series,
        credit_funding_effective=axis_bundle.credit_funding_effective,
        volume_liquidity_state=axis_bundle.volume_liquidity_state,
        volume_zscore_20d=(
            feature_store.volume_liquidity_v2.volume_zscore_20d
            if feature_store.volume_liquidity_v2 is not None
            else None
        ),
        gap_frequency_percentile_252d=volatility_v2.gap_frequency_percentile_252d,
        intraday_range_percentile_252d=volatility_v2.intraday_range_percentile_252d,
        hmm_top_state_prob=(
            feature_store.hmm.top_state_prob if feature_store.hmm is not None else None
        ),
        change_point_score=(
            feature_store.change_point.score
            if feature_store.change_point is not None
            else None
        ),
        cluster_id=(
            feature_store.clustering.cluster_id
            if feature_store.clustering is not None
            else None
        ),
    )

    outputs = build_transition_risk_outputs_by_date(
        sessions=sessions,
        trend_direction_active_by_date=axis_bundle.trend_direction.active_labels_by_date,
        trend_character_active_by_date=axis_bundle.trend_character.active_labels_by_date,
        volatility_state_active_by_date=axis_bundle.volatility_state.active_labels_by_date,
        breadth_state_active_by_date=axis_bundle.breadth_state.active_labels_by_date,
        close_by_date={
            day: float(value)
            for day, value in zip(sessions, close_series.to_numpy(), strict=True)
        },
        sma_50_by_date={
            day: None if pd.isna(value) else float(value)
            for day, value in zip(sessions, sma_50_series.to_numpy(), strict=True)
        },
        history=history,
        transition_score_inputs_by_date=transition_score_inputs_by_date,
        transition_score_config=transition_score_config,
        cooldown_window_days=transition_score_config.cooldown_window_days,
        state_confirmation_days=transition_score_config.state_confirmation_days,
        initial_active_state=transition_score_config.initial_active_state,
    )
    return _filter_transition_risk_outputs(
        outputs=outputs,
        output_sessions=requested_output_sessions,
    )


def _filter_transition_risk_outputs(
    *,
    outputs: dict[date, TransitionRiskOutput],
    output_sessions: list[date] | None,
) -> dict[date, TransitionRiskOutput]:
    if output_sessions is None:
        return outputs
    return {day: outputs[day] for day in output_sessions}


def _legacy_transition_score_config() -> TransitionScoreConfig:
    config = load_default_regime_config().transition_score
    if config is None:
        raise RuntimeError(
            "legacy transition_risk fallback requires default transition_score config"
        )
    return config


def _build_legacy_transition_score_inputs_by_date(
    *,
    sessions: list[date],
    event_calendar: dict[date, EventCalendarOutput],
    close: pd.Series,
    sma_50: pd.Series,
) -> dict[date, TransitionScoreInputs]:
    session_index = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
    close_values = close.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    sma50_values = sma_50.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )

    return {
        day: TransitionScoreInputs(
            realized_vol_short=float("nan"),
            realized_vol_long=float("nan"),
            pct_above_50dma=float("nan"),
            avg_pairwise_corr_percentile_504d=float("nan"),
            drawdown_252d=float("nan"),
            event_calendar_labels=event_calendar[day].matching_labels,
            spy_close=_optional_float(close_values[i]),
            spy_sma_50=_optional_float(sma50_values[i]),
        )
        for i, day in enumerate(sessions)
    }


def _build_transition_score_inputs_by_date(
    *,
    sessions: list[date],
    realized_vol_short: pd.Series,
    realized_vol_long: pd.Series,
    pct_above_50dma: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    largest_eigenvalue_share_percentile_504d: pd.Series,
    effective_rank_percentile_504d: pd.Series,
    absorption_ratio_top3: pd.Series,
    drawdown_252d: pd.Series,
    event_calendar: dict[date, EventCalendarOutput],
    close: pd.Series,
    sma_50: pd.Series,
    credit_funding_effective: dict[date, object] | None = None,
    volume_liquidity_state: dict[date, object] | None = None,
    volume_zscore_20d: pd.Series | None = None,
    gap_frequency_percentile_252d: pd.Series | None = None,
    intraday_range_percentile_252d: pd.Series | None = None,
    hmm_top_state_prob: pd.Series | None = None,
    change_point_score: pd.Series | None = None,
    cluster_id: pd.Series | None = None,
) -> dict[date, TransitionScoreInputs]:
    """Materialise the per-session v2 §4.2 score inputs for every NYSE session.

    Reindexes each input series ONCE against the session DatetimeIndex
    then iterates over numpy arrays — avoids the per-session ``.loc[ts]``
    lookup pattern. Missing index entries surface as NaN through
    ``.reindex`` and propagate to the
    ``compose_transition_score_for_session`` cold-start guard.
    """
    session_index = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions])
    rvs = realized_vol_short.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    rvl = realized_vol_long.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    pct50 = pct_above_50dma.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    corr = avg_pairwise_corr_percentile_504d.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    largest = largest_eigenvalue_share_percentile_504d.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    eff_rank = effective_rank_percentile_504d.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    absorption = absorption_ratio_top3.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    dd252 = drawdown_252d.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    close_values = close.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    sma50_values = sma_50.reindex(session_index).to_numpy(
        dtype=float, na_value=float("nan")
    )
    nan_float = np.full(len(sessions), np.nan, dtype=float)
    volume_z = (
        volume_zscore_20d.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
        if volume_zscore_20d is not None
        else nan_float
    )
    gap_pct = (
        gap_frequency_percentile_252d.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
        if gap_frequency_percentile_252d is not None
        else nan_float
    )
    intraday_pct = (
        intraday_range_percentile_252d.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
        if intraday_range_percentile_252d is not None
        else nan_float
    )

    # v2 §6.1 — bulk-reindex both `top_state_prob[t]` and
    # `top_state_prob[t-5]`. The 5-NYSE-session offset is materialized by
    # shifting the SOURCE series before reindexing so each session t maps
    # to the value at session t-5 (or NaN if absent).
    if hmm_top_state_prob is not None:
        hmm_now = hmm_top_state_prob.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
        hmm_5d_ago = (
            hmm_top_state_prob.shift(5)
            .reindex(session_index)
            .to_numpy(dtype=float, na_value=float("nan"))
        )
    else:
        hmm_now = nan_float
        hmm_5d_ago = nan_float

    # documented implementation decision — change_point.score (5-session rolling max of
    # recent short-run BOCPD posterior mass; already in [0, 1]).
    if change_point_score is not None:
        cp = change_point_score.reindex(session_index).to_numpy(
            dtype=float, na_value=float("nan")
        )
    else:
        cp = nan_float

    if cluster_id is not None:
        cluster_now = cluster_id.reindex(session_index).to_numpy()
        cluster_5d_ago = cluster_id.shift(5).reindex(session_index).to_numpy()
    else:
        nan_cluster = np.full(len(sessions), pd.NA, dtype=object)
        cluster_now = nan_cluster
        cluster_5d_ago = nan_cluster

    out: dict[date, TransitionScoreInputs] = {}
    for i, day in enumerate(sessions):
        out[day] = TransitionScoreInputs(
            realized_vol_short=float(rvs[i]),
            realized_vol_long=float(rvl[i]),
            pct_above_50dma=float(pct50[i]),
            avg_pairwise_corr_percentile_504d=float(corr[i]),
            drawdown_252d=float(dd252[i]),
            event_calendar_labels=event_calendar[day].matching_labels,
            spy_close=_optional_float(close_values[i]),
            spy_sma_50=_optional_float(sma50_values[i]),
            largest_eigenvalue_share_percentile_504d=_optional_float(largest[i]),
            effective_rank_percentile_504d=_optional_float(eff_rank[i]),
            absorption_ratio_top3=_optional_float(absorption[i]),
            credit_funding_label=_active_label_for_day(credit_funding_effective, day),
            volume_liquidity_label=_active_label_for_day(volume_liquidity_state, day),
            volume_zscore_20d=_optional_float(volume_z[i]),
            gap_frequency_percentile_252d=_optional_float(gap_pct[i]),
            intraday_range_percentile_252d=_optional_float(intraday_pct[i]),
            hmm_top_state_prob_now=_optional_float(hmm_now[i]),
            hmm_top_state_prob_5d_ago=_optional_float(hmm_5d_ago[i]),
            change_point_score=_optional_float(cp[i]),
            cluster_id_now=_optional_int(cluster_now[i]),
            cluster_id_5d_ago=_optional_int(cluster_5d_ago[i]),
        )
    return out


def _optional_float(value: object) -> float | None:
    # Guard pd.NA first: float(pd.NA) raises TypeError because NAType is
    # not a real number. Numpy NaN is fine — float(nan) returns nan and
    # pd.isna(nan) is True. Mirrors _optional_int's pd.isna-first shape.
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _active_label_for_day(outputs: dict[date, object] | None, day: date) -> str | None:
    if outputs is None:
        return None
    output = outputs.get(day)
    if output is None:
        return None
    label = getattr(output, "active_label", None)
    if label is None:
        return None
    status = getattr(output, "classification_status", "classified")
    if status != "classified":
        return None
    return str(label)


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
    transition_score_inputs_by_date: dict[date, TransitionScoreInputs],
    transition_score_config: TransitionScoreConfig | None,
    cooldown_window_days: int = 5,
    state_confirmation_days: dict[str, int] | None = None,
    initial_active_state: str | None = None,
) -> dict[date, TransitionRiskOutput]:
    index = pd.Index(sessions)
    trend_direction_active = pd.Series(
        [trend_direction_active_by_date[day] for day in sessions], index=index
    )
    trend_character_active = pd.Series(
        [trend_character_active_by_date[day] for day in sessions], index=index
    )
    volatility_state_active = pd.Series(
        [volatility_state_active_by_date[day] for day in sessions], index=index
    )
    breadth_state_active = pd.Series(
        [breadth_state_active_by_date[day] for day in sessions], index=index
    )
    close = pd.Series(
        [close_by_date[day] for day in sessions], index=index, dtype="float64"
    )
    sma_50 = pd.Series(
        [sma_50_by_date[day] for day in sessions], index=index, dtype="float64"
    )
    prior_bear = pd.Series(
        [history.prior_bear_by_date[day] for day in sessions], index=index, dtype="bool"
    )
    days_since_axis_switch = pd.Series(
        [history.days_since_axis_switch_by_date[day] for day in sessions],
        index=index,
        dtype="float64",
    )

    recovery_attempt = trend_character_active.eq("recovery_attempt") | (
        prior_bear
        & close.gt(sma_50)
        & breadth_state_active.isin(["recovery_breadth", "healthy_breadth"])
    )
    volatility_crisis = volatility_state_active.eq("crisis_vol")
    volatility_high_or_crisis = volatility_state_active.isin(["high_vol", "crisis_vol"])
    breadth_stressed = breadth_state_active.isin(
        ["weak_breadth", "narrowing_breadth", "divergent_fragile", "unknown"]
    )
    post_switch_cooldown = (
        days_since_axis_switch.notna()
        & days_since_axis_switch.le(cooldown_window_days)
        & ~volatility_crisis
    )
    insufficient_data = (
        trend_direction_active.eq("unknown")
        | trend_character_active.eq("unknown")
        | volatility_state_active.eq("unknown")
        | breadth_state_active.eq("unknown")
    )

    # Materialize each Series as a numpy array once. The downstream loop
    # then indexes by integer position, avoiding O(n) repeated label lookups
    # via .loc[day] across n sessions. Mirrors the bulk-reindex pattern
    # already applied to the score inputs in
    # _build_transition_score_inputs_by_date.
    trend_direction_arr = trend_direction_active.to_numpy()
    volatility_state_arr = volatility_state_active.to_numpy()
    breadth_state_arr = breadth_state_active.to_numpy()
    recovery_attempt_arr = recovery_attempt.to_numpy(dtype=bool)
    volatility_crisis_arr = volatility_crisis.to_numpy(dtype=bool)
    volatility_high_or_crisis_arr = volatility_high_or_crisis.to_numpy(dtype=bool)
    breadth_stressed_arr = breadth_stressed.to_numpy(dtype=bool)
    post_switch_cooldown_arr = post_switch_cooldown.to_numpy(dtype=bool)
    insufficient_data_arr = insufficient_data.to_numpy(dtype=bool)

    selection_config = transition_score_config or _legacy_transition_score_config()
    raw_outputs: dict[date, TransitionRiskOutput] = {}
    for i, day in enumerate(sessions):
        switch_days = history.days_since_axis_switch_by_date[day]
        inputs = transition_score_inputs_by_date[day]
        if transition_score_config is None:
            composed = ComposedTransitionScore(
                score=0.0,
                interpretation="stable",
                components={},
                macro_event_labels=inputs.event_calendar_labels,
            )
        elif bool(insufficient_data_arr[i]):
            composed = ComposedTransitionScore(
                score=None,
                interpretation=None,
                components=None,
                missing_components=("axis_data_quality",),
            )
        else:
            composed = compose_transition_score_for_session(
                realized_vol_short=inputs.realized_vol_short,
                realized_vol_long=inputs.realized_vol_long,
                pct_above_50dma=inputs.pct_above_50dma,
                avg_pairwise_corr_percentile_504d=(
                    inputs.avg_pairwise_corr_percentile_504d
                ),
                drawdown_252d=inputs.drawdown_252d,
                event_calendar_labels=inputs.event_calendar_labels,
                spy_close=inputs.spy_close,
                spy_sma_50=inputs.spy_sma_50,
                largest_eigenvalue_share_percentile_504d=(
                    inputs.largest_eigenvalue_share_percentile_504d
                ),
                effective_rank_percentile_504d=inputs.effective_rank_percentile_504d,
                absorption_ratio_top3=inputs.absorption_ratio_top3,
                credit_funding_label=inputs.credit_funding_label,
                volume_liquidity_label=inputs.volume_liquidity_label,
                volume_zscore_20d=inputs.volume_zscore_20d,
                gap_frequency_percentile_252d=inputs.gap_frequency_percentile_252d,
                intraday_range_percentile_252d=inputs.intraday_range_percentile_252d,
                hmm_top_state_prob_now=inputs.hmm_top_state_prob_now,
                hmm_top_state_prob_5d_ago=inputs.hmm_top_state_prob_5d_ago,
                change_point_score=inputs.change_point_score,
                cluster_id_now=inputs.cluster_id_now,
                cluster_id_5d_ago=inputs.cluster_id_5d_ago,
                config=transition_score_config,
            )
        components = composed.components or {}
        overrides = selection_config.overrides
        credit_stressed = (
            components.get("credit_stress", 0.0) >= overrides.credit_stress
        )
        correlation_stressed = (
            components.get("correlation_fragility", 0.0)
            >= overrides.correlation_fragility
        )
        macro_elevated = components.get("macro_event", 0.0) >= overrides.macro_event_min
        score_elevated = (
            composed.score is not None
            and composed.score >= overrides.score_elevated_min
        )
        # Absolute old-behavior emergency override: crisis_vol alone is enough.
        crisis = bool(volatility_crisis_arr[i])
        bear_stress = (
            trend_direction_arr[i] == "bear"
            and bool(volatility_high_or_crisis_arr[i])
            and (bool(breadth_stressed_arr[i]) or credit_stressed)
        )
        fragile_bull = trend_direction_arr[i] == "bull" and (
            breadth_state_arr[i] == "divergent_fragile"
            or correlation_stressed
            or credit_stressed
        )
        # Preserve the old V2 sideways-stress shape, mapped to watch instead of
        # a separate final state.
        sideways_stress = (
            trend_direction_arr[i] == "sideways"
            and volatility_state_arr[i] == "high_vol"
            and breadth_state_arr[i]
            in {"weak_breadth", "narrowing_breadth", "divergent_fragile"}
        )
        event_transition_watch = bool(
            macro_elevated
            and score_elevated
            and components.get("macro_event", 0.0)
            >= max(
                (value for key, value in components.items() if key != "macro_event"),
                default=0.0,
            )
        )
        output = compose_transition_risk_output(
            score=composed,
            primary_driver_min=overrides.primary_driver_min,
            flags=TransitionRuleFlags(
                crisis=crisis,
                bear_stress=bear_stress,
                fragile_bull=fragile_bull,
                recovery_attempt=bool(recovery_attempt_arr[i]),
                sideways_stress=sideways_stress,
                event_transition_watch=event_transition_watch,
                post_switch_cooldown=bool(post_switch_cooldown_arr[i]),
                insufficient_data=bool(insufficient_data_arr[i]),
                stable_changed_today=history.stable_changed_by_date[day],
                days_since_axis_switch=switch_days,
                axis_switch_count=history.axis_switch_count_by_date[day],
                recent_axis_switch_count=history.recent_axis_switch_count_by_date[day],
            ),
        )
        if transition_score_config is None:
            output = output.model_copy(
                update={
                    "score": None,
                    "score_components": None,
                    "primary_drivers": [],
                }
            )
        raw_outputs[day] = output
    return _apply_transition_state_debounce(
        sessions=sessions,
        raw_outputs=raw_outputs,
        state_confirmation_days=state_confirmation_days
        or selection_config.state_confirmation_days,
        initial_active_state=initial_active_state,
    )


def _apply_transition_state_debounce(
    *,
    sessions: list[date],
    raw_outputs: dict[date, TransitionRiskOutput],
    state_confirmation_days: dict[str, int],
    initial_active_state: str | None = None,
) -> dict[date, TransitionRiskOutput]:
    outputs: dict[date, TransitionRiskOutput] = {}
    # Default (initial_active_state=None) preserves the historical
    # backfill behavior where the first session's raw state is accepted
    # immediately. Setting initial_active_state seeds the debounce so that
    # the first session must also clear its configured confirmation window
    # — useful for live streaming, where no prior session can bootstrap.
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

    axis_changed = stable_frame.ne(stable_frame.shift(1))
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
    axis_switch_count_by_date = {
        day: int(value) for day, value in axis_switch_count.items()
    }
    recent_axis_switch_count = (
        axis_switch_count.rolling(window=5, min_periods=1).sum().astype("int64")
    )
    recent_axis_switch_count_by_date = {
        day: int(value) for day, value in recent_axis_switch_count.items()
    }
    days_since_axis_switch_by_date = {
        day: None if pd.isna(value) else int(value)
        for day, value in days_since_axis_switch.items()
    }
    prior_bear_by_date = {day: bool(value) for day, value in prior_bear.items()}
    return TransitionRiskHistory(
        stable_changed_by_date=stable_changed_by_date,
        days_since_axis_switch_by_date=days_since_axis_switch_by_date,
        axis_switch_count_by_date=axis_switch_count_by_date,
        recent_axis_switch_count_by_date=recent_axis_switch_count_by_date,
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
        missing_sessions = [
            session_index[idx].date().isoformat()
            for idx, pos in enumerate(positions)
            if pos < 0
        ][:5]
        raise ValueError(
            f"{series_name} index is not exactly aligned to NYSE sessions used by transition-risk computation. "
            f"Missing exact matches for sessions: {missing_sessions}"
        )
    return pd.Series(
        source.to_numpy()[positions], index=session_index, name=source.name
    )
