from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import get_type_hints

import pandas as pd
import pytest
from pydantic import ValidationError

from regime_detection.config import load_default_regime_config
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    EventCalendarOutput,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
)
from regime_detection.axis_series import AxisSeriesBundle, AxisSeriesResult
from regime_detection.feature_store import FeatureStore
from regime_detection.transition_risk import (
    TransitionRuleFlags,
    compose_transition_risk_output,
)
from regime_detection.transition_risk_series import (
    EventCalendarLabel,
    TransitionRiskHistory,
    TransitionScoreInputs,
    _build_transition_score_inputs_by_date,
    build_transition_risk_outputs_by_date,
    build_transition_risk_series,
    build_transition_risk_history,
)
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compose_transition_score_for_session,
)


def _flags(**overrides: object) -> TransitionRuleFlags:
    kwargs = {
        "crisis": False,
        "bear_stress": False,
        "fragile_bull": False,
        "recovery_attempt": False,
        "sideways_stress": False,
        "event_transition_watch": False,
        "post_switch_cooldown": False,
        "insufficient_data": False,
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "axis_switch_count": 0,
        "recent_axis_switch_count": 0,
    }
    kwargs.update(overrides)
    return TransitionRuleFlags(**kwargs)


def test_transition_risk_uses_score_band_as_base_state() -> None:
    output = compose_transition_risk_output(
        score=ComposedTransitionScore(
            score=0.64,
            interpretation="transition_warning",
            components={
                "breadth_deterioration": 0.80,
                "volatility_acceleration": 0.60,
                "trend_break": 0.20,
            },
        ),
        flags=_flags(),
    )

    assert output.state == "transition_warning"
    assert output.primary_drivers == [
        "breadth_deterioration",
        "volatility_acceleration",
    ]
    assert output.triggered_rules == []
    assert output.data_quality.status == "ok"


def test_transition_risk_composer_emits_typed_evidence_payload() -> None:
    output = compose_transition_risk_output(
        score=ComposedTransitionScore(
            score=0.64,
            interpretation="transition_warning",
            components={"breadth_deterioration": 0.80},
            macro_event_labels=("fed_week",),
        ),
        flags=_flags(stable_changed_today=True, axis_switch_count=2),
    )

    assert type(output.evidence) is TransitionRiskEvidencePayload
    assert output.evidence.triggered_rules == []
    assert output.evidence.stable_changed_today is True
    assert output.evidence.axis_switch_count == 2
    assert output.evidence.macro_event_labels == ["fed_week"]


def test_transition_risk_hard_override_wins_over_generic_score_state() -> None:
    output = compose_transition_risk_output(
        score=ComposedTransitionScore(
            score=0.82,
            interpretation="high",
            components={"trend_break": 1.0},
        ),
        flags=_flags(bear_stress=True),
    )

    assert output.state == "bear_stress"
    assert output.triggered_rules == ["bear_stress"]


def test_transition_risk_crisis_override_wins_over_insufficient_axis_data() -> None:
    output = compose_transition_risk_output(
        score=ComposedTransitionScore(
            score=0.10,
            interpretation="stable",
            components={"trend_break": 0.10},
        ),
        flags=_flags(crisis=True, insufficient_data=True),
    )

    assert output.state == "crisis"
    assert output.triggered_rules == ["crisis", "insufficient_data"]


def test_transition_risk_watch_rules_win_over_insufficient_axis_data() -> None:
    score = ComposedTransitionScore(
        score=0.10,
        interpretation="stable",
        components={"trend_break": 0.10},
    )

    event_watch = compose_transition_risk_output(
        score=score,
        flags=_flags(event_transition_watch=True, insufficient_data=True),
    )
    cooldown_watch = compose_transition_risk_output(
        score=score,
        flags=_flags(post_switch_cooldown=True, insufficient_data=True),
    )

    assert event_watch.state == "watch"
    assert event_watch.triggered_rules == [
        "event_transition_watch",
        "insufficient_data",
    ]
    assert cooldown_watch.state == "watch"
    assert cooldown_watch.triggered_rules == [
        "post_switch_cooldown",
        "insufficient_data",
    ]


def test_transition_risk_missing_score_becomes_insufficient_data() -> None:
    output = compose_transition_risk_output(
        score=ComposedTransitionScore(score=None, interpretation=None, components=None),
        flags=_flags(),
    )

    assert output.state == "insufficient_data"
    assert output.score is None
    assert output.data_quality.status == "insufficient_history"
    assert output.classification_status == "insufficient_history"


def test_transition_risk_evidence_preserves_dict_access_and_dump_shape() -> None:
    output = TransitionRiskOutput(
        state="stable",
        evidence={
            "triggered_rules": [],
            "stable_changed_today": False,
            "days_since_axis_switch": None,
            "axis_switch_count": 0,
            "recent_axis_switch_count": 0,
            "macro_event_labels": [],
        },
        score=0.10,
        score_components={"trend_break": 0.10},
        primary_drivers=[],
        data_quality={"status": "ok"},
    )

    expected = {
        "triggered_rules": [],
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "axis_switch_count": 0,
        "recent_axis_switch_count": 0,
        "macro_event_labels": [],
    }
    assert output.evidence["triggered_rules"] == []
    assert output.evidence.get("stable_changed_today") is False
    assert output.evidence.get("missing", "fallback") == "fallback"
    assert output.evidence == expected
    assert output.evidence.model_dump() == expected


@pytest.mark.parametrize("bad_key", ["warning_active", "legacy_warning_list", "reason"])
def test_transition_risk_evidence_rejects_unknown_keys(bad_key: str) -> None:
    with pytest.raises(ValidationError, match=bad_key):
        TransitionRiskOutput(
            state="stable",
            evidence={
                "triggered_rules": [],
                "stable_changed_today": False,
                "days_since_axis_switch": None,
                "axis_switch_count": 0,
                "recent_axis_switch_count": 0,
                bad_key: "unexpected",
            },
            score=0.10,
            score_components={"trend_break": 0.10},
            data_quality={"status": "ok"},
        )


def test_build_transition_score_inputs_returns_typed_optional_hmm_and_change_point_values() -> (
    None
):
    sessions = [date(2024, 1, day) for day in range(2, 10)]
    index = pd.DatetimeIndex(sessions)

    inputs_by_date = _build_transition_score_inputs_by_date(
        sessions=sessions,
        realized_vol_short=pd.Series([12.0] * len(sessions), index=index),
        realized_vol_long=pd.Series([10.0] * len(sessions), index=index),
        pct_above_50dma=pd.Series([0.45] * len(sessions), index=index),
        avg_pairwise_corr_percentile_504d=pd.Series(
            [0.60] * len(sessions), index=index
        ),
        largest_eigenvalue_share_percentile_504d=pd.Series(
            [0.50] * len(sessions), index=index
        ),
        effective_rank_percentile_504d=pd.Series([0.50] * len(sessions), index=index),
        absorption_ratio_top3=pd.Series([0.50] * len(sessions), index=index),
        drawdown_252d=pd.Series([-0.10] * len(sessions), index=index),
        close=pd.Series([100.0] * len(sessions), index=index),
        sma_50=pd.Series([95.0] * len(sessions), index=index),
        event_calendar={
            day: EventCalendarOutput(
                primary_label="normal_calendar",
                matching_labels=("normal_calendar",),
                evidence={"selection_method": "precedence"},
            )
            for day in sessions
        },
        hmm_top_state_prob=pd.Series(
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.70, 0.75],
            index=index,
        ),
        change_point_score=pd.Series([0.20] * len(sessions), index=index),
        cluster_id=pd.Series([1, 1, 1, 1, 1, 2, 2, 2], index=index),
    )

    first_day = sessions[0]
    shifted_day = sessions[5]
    assert isinstance(inputs_by_date[first_day], TransitionScoreInputs)
    assert inputs_by_date[first_day].hmm_top_state_prob_now == pytest.approx(0.10)
    assert pd.isna(inputs_by_date[first_day].hmm_top_state_prob_5d_ago)
    assert inputs_by_date[first_day].change_point_score == pytest.approx(0.20)
    assert inputs_by_date[first_day].cluster_id_now == 1
    assert inputs_by_date[first_day].cluster_id_5d_ago is None
    assert inputs_by_date[shifted_day].hmm_top_state_prob_now == pytest.approx(0.65)
    assert inputs_by_date[shifted_day].hmm_top_state_prob_5d_ago == pytest.approx(0.10)
    assert inputs_by_date[shifted_day].cluster_id_now == 2
    assert inputs_by_date[shifted_day].cluster_id_5d_ago == 1


def test_transition_risk_series_matches_direct_score_composer() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    session = date(2024, 1, 9)
    score_inputs = TransitionScoreInputs(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_labels=("normal_calendar",),
        hmm_top_state_prob_now=0.70,
        hmm_top_state_prob_5d_ago=0.30,
        change_point_score=0.50,
        cluster_id_now=1,
        cluster_id_5d_ago=1,
    )

    outputs = build_transition_risk_outputs_by_date(
        sessions=[session],
        trend_direction_active_by_date={session: "bull"},
        trend_character_active_by_date={session: "trending"},
        volatility_state_active_by_date={session: "normal_vol"},
        breadth_state_active_by_date={session: "healthy_breadth"},
        close_by_date={session: 100.0},
        sma_50_by_date={session: 95.0},
        history=TransitionRiskHistory(
            stable_changed_by_date={session: False},
            days_since_axis_switch_by_date={session: None},
            axis_switch_count_by_date={session: 0},
            recent_axis_switch_count_by_date={session: 0},
            prior_bear_by_date={session: False},
        ),
        transition_score_inputs_by_date={session: score_inputs},
        transition_score_config=cfg,
    )
    expected = compose_transition_score_for_session(
        realized_vol_short=score_inputs.realized_vol_short,
        realized_vol_long=score_inputs.realized_vol_long,
        pct_above_50dma=score_inputs.pct_above_50dma,
        avg_pairwise_corr_percentile_504d=score_inputs.avg_pairwise_corr_percentile_504d,
        drawdown_252d=score_inputs.drawdown_252d,
        event_calendar_labels=score_inputs.event_calendar_labels,
        hmm_top_state_prob_now=score_inputs.hmm_top_state_prob_now,
        hmm_top_state_prob_5d_ago=score_inputs.hmm_top_state_prob_5d_ago,
        change_point_score=score_inputs.change_point_score,
        cluster_id_now=score_inputs.cluster_id_now,
        cluster_id_5d_ago=score_inputs.cluster_id_5d_ago,
        config=cfg,
    )

    assert outputs[session].score == expected.score
    assert outputs[session].score_components == expected.components
    assert outputs[session].state == "weakening"


def test_transition_risk_evidence_preserves_macro_event_matching_labels() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    session = date(2024, 1, 9)
    score_inputs = TransitionScoreInputs(
        realized_vol_short=10.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.80,
        avg_pairwise_corr_percentile_504d=0.0,
        drawdown_252d=0.0,
        event_calendar_labels=(
            "earnings_season",
            "fed_week",
            "expiry_week",
            "cpi_week",
        ),
        spy_close=100.0,
        spy_sma_50=100.0,
        largest_eigenvalue_share_percentile_504d=0.0,
        effective_rank_percentile_504d=1.0,
        absorption_ratio_top3=0.50,
        credit_funding_label="credit_calm",
        volume_liquidity_label="normal_volume",
        volume_zscore_20d=1.0,
        gap_frequency_percentile_252d=0.0,
        intraday_range_percentile_252d=0.0,
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
        cluster_id_now=1,
        cluster_id_5d_ago=1,
    )

    outputs = build_transition_risk_outputs_by_date(
        sessions=[session],
        trend_direction_active_by_date={session: "bull"},
        trend_character_active_by_date={session: "trending"},
        volatility_state_active_by_date={session: "normal_vol"},
        breadth_state_active_by_date={session: "healthy_breadth"},
        close_by_date={session: 100.0},
        sma_50_by_date={session: 100.0},
        history=TransitionRiskHistory(
            stable_changed_by_date={session: False},
            days_since_axis_switch_by_date={session: None},
            axis_switch_count_by_date={session: 0},
            recent_axis_switch_count_by_date={session: 0},
            prior_bear_by_date={session: False},
        ),
        transition_score_inputs_by_date={session: score_inputs},
        transition_score_config=cfg,
    )

    assert outputs[session].evidence.macro_event_labels == ["fed_week", "cpi_week"]


def test_transition_score_inputs_event_calendar_labels_are_closed_type() -> None:
    assert (
        get_type_hints(TransitionScoreInputs)["event_calendar_labels"]
        == tuple[EventCalendarLabel, ...]
    )
    with pytest.raises(ValueError, match="unknown event_calendar_labels"):
        TransitionScoreInputs(
            realized_vol_short=12.0,
            realized_vol_long=10.0,
            pct_above_50dma=0.45,
            avg_pairwise_corr_percentile_504d=0.60,
            drawdown_252d=-0.10,
            event_calendar_labels=("vendor_changed_name",),  # type: ignore[arg-type]
        )


def test_transition_score_missing_model_evidence_marks_component_missing() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None

    out = compose_transition_score_for_session(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_labels=("normal_calendar",),
        credit_funding_label="credit_calm",
        volume_liquidity_label="normal_volume",
        config=cfg,
    )

    assert out.score is not None
    assert out.components is not None
    assert "model_instability" not in out.components
    assert out.missing_components == ("model_instability",)


def test_transition_risk_series_degrades_when_model_evidence_is_cold_start() -> None:
    session = date(2024, 1, 2)
    index = pd.DatetimeIndex([pd.Timestamp(session)])
    context = MarketContext.model_construct(
        end_date=session,
        config=load_default_regime_config(),
        sessions=(session,),
        spy_ohlcv=pd.DataFrame({"close": [100.0]}, index=index),
    )
    feature_store = FeatureStore.model_construct(
        spy_index=index,
        sma_50=pd.Series([100.0], index=index),
        volatility_state_v2=SimpleNamespace(
            realized_vol_short=pd.Series([10.0], index=index),
            realized_vol_long=pd.Series([10.0], index=index),
            gap_frequency_percentile_252d=pd.Series([0.0], index=index),
            intraday_range_percentile_252d=pd.Series([0.0], index=index),
        ),
        breadth_state_v2=SimpleNamespace(
            pct_above_50dma=pd.Series([0.5], index=index),
        ),
        network_fragility=SimpleNamespace(
            avg_pairwise_corr_percentile_504d=pd.Series([0.0], index=index),
            largest_eigenvalue_share_percentile_504d=pd.Series([0.0], index=index),
            effective_rank_percentile_504d=pd.Series([1.0], index=index),
            absorption_ratio_top3=pd.Series([0.5], index=index),
        ),
        trend_direction_v2=SimpleNamespace(
            drawdown_252d=pd.Series([0.0], index=index),
        ),
        volume_liquidity_v2=None,
        hmm=None,
        change_point=None,
        clustering=None,
    )
    axis_bundle = AxisSeriesBundle(
        trend_direction=_axis_result([session], "bull"),
        trend_character=_axis_result([session], "trending"),
        volatility_state=_axis_result([session], "low_vol"),
        breadth_state=_axis_result([session], "healthy_breadth"),
        event_calendar=_event_calendar([session]),
    )

    outputs = build_transition_risk_series(
        context=context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    assert outputs[session].score is not None
    assert outputs[session].score_components is not None
    assert "model_instability" not in outputs[session].score_components


def test_transition_risk_state_debounces_soft_state_changes() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    sessions = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]
    outputs = build_transition_risk_outputs_by_date(
        sessions=sessions,
        trend_direction_active_by_date={day: "sideways" for day in sessions},
        trend_character_active_by_date={day: "trending" for day in sessions},
        volatility_state_active_by_date={day: "normal_vol" for day in sessions},
        breadth_state_active_by_date={day: "healthy_breadth" for day in sessions},
        close_by_date={day: 100.0 for day in sessions},
        sma_50_by_date={day: 100.0 for day in sessions},
        history=TransitionRiskHistory(
            stable_changed_by_date={day: False for day in sessions},
            days_since_axis_switch_by_date={day: None for day in sessions},
            axis_switch_count_by_date={day: 0 for day in sessions},
            recent_axis_switch_count_by_date={day: 0 for day in sessions},
            prior_bear_by_date={day: False for day in sessions},
        ),
        transition_score_inputs_by_date={
            sessions[0]: TransitionScoreInputs(
                realized_vol_short=10.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.50,
                avg_pairwise_corr_percentile_504d=0.0,
                drawdown_252d=0.0,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_stress",
                volume_liquidity_label="liquidity_gap_behavior",
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            ),
            sessions[1]: TransitionScoreInputs(
                realized_vol_short=15.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.20,
                avg_pairwise_corr_percentile_504d=1.0,
                drawdown_252d=-0.15,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_stress",
                volume_liquidity_label="liquidity_gap_behavior",
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            ),
            sessions[2]: TransitionScoreInputs(
                realized_vol_short=15.0,
                realized_vol_long=10.0,
                pct_above_50dma=0.20,
                avg_pairwise_corr_percentile_504d=1.0,
                drawdown_252d=-0.15,
                event_calendar_labels=("normal_calendar",),
                credit_funding_label="credit_stress",
                volume_liquidity_label="liquidity_gap_behavior",
                hmm_top_state_prob_now=0.50,
                hmm_top_state_prob_5d_ago=0.50,
                change_point_score=0.0,
                cluster_id_now=1,
                cluster_id_5d_ago=1,
            ),
        },
        transition_score_config=cfg,
    )

    assert outputs[sessions[0]].state == "stable"
    assert outputs[sessions[1]].state == "stable"
    assert outputs[sessions[1]].triggered_rules == ["state_confirmation_pending"]
    assert outputs[sessions[2]].state == "high_transition_risk"


def _axis_result(sessions: list[date], label: str) -> AxisSeriesResult:
    return AxisSeriesResult(
        outputs_by_date={},
        stable_labels_by_date={day: label for day in sessions},
        active_labels_by_date={day: label for day in sessions},
    )


def _event_calendar(sessions: list[date]) -> dict[date, EventCalendarOutput]:
    return {
        day: EventCalendarOutput(
            primary_label="normal_calendar",
            matching_labels=("normal_calendar",),
            evidence={},
        )
        for day in sessions
    }


def test_v1_transition_risk_fallback_preserves_flag_only_stable_state() -> None:
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in sessions])
    config = load_default_regime_config().model_copy(update={"transition_score": None})
    context = MarketContext.model_construct(
        end_date=sessions[-1],
        config=config,
        sessions=tuple(sessions),
        spy_ohlcv=pd.DataFrame({"close": [100.0, 101.0]}, index=index),
    )
    feature_store = FeatureStore.model_construct(
        spy_index=index,
        sma_50=pd.Series([99.0, 100.0], index=index),
    )
    axis_bundle = AxisSeriesBundle(
        trend_direction=_axis_result(sessions, "bull"),
        trend_character=_axis_result(sessions, "trending"),
        volatility_state=_axis_result(sessions, "low_vol"),
        breadth_state=_axis_result(sessions, "healthy_breadth"),
        event_calendar=_event_calendar(sessions),
    )

    outputs = build_transition_risk_series(
        context=context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    assert outputs[sessions[-1]].state == "stable"
    assert outputs[sessions[-1]].score is None
    assert outputs[sessions[-1]].score_components is None
    assert outputs[sessions[-1]].classification_status == "classified"


def test_v1_transition_risk_requires_no_model_evidence() -> None:
    # F-041 / ADR 0020: the V1-config path (transition_score=None) emits a
    # strategy-consumable transition_risk.state WITHOUT any V2 model evidence.
    # This is the deliberate inverse of
    # test_transition_risk_series_requires_model_evidence_when_score_enabled:
    # V1 must not raise on absent HMM / change-point / clustering, because V1
    # §10 strategy response depends on transition_risk.state existing, while the
    # forbidden V2 weighted score (score / score_components) stays absent.
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in sessions])
    config = load_default_regime_config().model_copy(update={"transition_score": None})
    assert config.transition_score is None
    context = MarketContext.model_construct(
        end_date=sessions[-1],
        config=config,
        sessions=tuple(sessions),
        spy_ohlcv=pd.DataFrame({"close": [100.0, 101.0]}, index=index),
    )
    # FeatureStore deliberately omits hmm / change_point / clustering.
    feature_store = FeatureStore.model_construct(
        spy_index=index,
        sma_50=pd.Series([99.0, 100.0], index=index),
    )
    assert feature_store.hmm is None
    assert feature_store.change_point is None
    assert feature_store.clustering is None
    axis_bundle = AxisSeriesBundle(
        trend_direction=_axis_result(sessions, "bull"),
        trend_character=_axis_result(sessions, "trending"),
        volatility_state=_axis_result(sessions, "low_vol"),
        breadth_state=_axis_result(sessions, "healthy_breadth"),
        event_calendar=_event_calendar(sessions),
    )

    # Must NOT raise the "model evidence" RuntimeError that the V2 path raises.
    outputs = build_transition_risk_series(
        context=context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    final = outputs[sessions[-1]]
    # State is emitted for strategy response; the V2 weighted score stays absent.
    assert final.state is not None
    assert final.score is None
    assert final.score_components is None


def test_transition_risk_output_sessions_debounce_uses_full_session_history() -> None:
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in sessions])
    model_evidence_index = pd.DatetimeIndex(
        [
            pd.Timestamp(date(2023, 12, 22)),
            pd.Timestamp(date(2023, 12, 26)),
            pd.Timestamp(date(2023, 12, 27)),
            pd.Timestamp(date(2023, 12, 28)),
            pd.Timestamp(date(2023, 12, 29)),
            *index,
        ]
    )
    config = load_default_regime_config()
    context = MarketContext.model_construct(
        end_date=sessions[-1],
        config=config,
        sessions=tuple(sessions),
        spy_ohlcv=pd.DataFrame({"close": [100.0, 99.0, 98.0]}, index=index),
    )
    feature_store = FeatureStore.model_construct(
        spy_index=index,
        sma_50=pd.Series([100.0, 100.0, 100.0], index=index),
        volatility_state_v2=SimpleNamespace(
            realized_vol_short=pd.Series([10.0, 15.0, 15.0], index=index),
            realized_vol_long=pd.Series([10.0, 10.0, 10.0], index=index),
            gap_frequency_percentile_252d=pd.Series([0.0, 0.0, 0.0], index=index),
            intraday_range_percentile_252d=pd.Series([0.0, 0.0, 0.0], index=index),
        ),
        breadth_state_v2=SimpleNamespace(
            pct_above_50dma=pd.Series([0.5, 0.2, 0.2], index=index),
        ),
        network_fragility=SimpleNamespace(
            avg_pairwise_corr_percentile_504d=pd.Series([0.0, 1.0, 1.0], index=index),
            largest_eigenvalue_share_percentile_504d=pd.Series(
                [0.0, 0.0, 0.0], index=index
            ),
            effective_rank_percentile_504d=pd.Series([0.0, 0.0, 0.0], index=index),
            absorption_ratio_top3=pd.Series([0.0, 0.0, 0.0], index=index),
        ),
        trend_direction_v2=SimpleNamespace(
            drawdown_252d=pd.Series([0.0, -0.15, -0.15], index=index),
        ),
        volume_liquidity_v2=None,
        hmm=SimpleNamespace(
            top_state_prob=pd.Series(
                [0.5] * len(model_evidence_index), index=model_evidence_index
            ),
        ),
        change_point=SimpleNamespace(
            score=pd.Series(
                [0.0] * len(model_evidence_index), index=model_evidence_index
            ),
        ),
        clustering=SimpleNamespace(
            cluster_id=pd.Series(
                [1] * len(model_evidence_index), index=model_evidence_index
            ),
        ),
    )
    classified_credit_stress = SimpleNamespace(
        active_label="credit_stress",
        classification_status="classified",
    )
    classified_liquidity_gap = SimpleNamespace(
        active_label="liquidity_gap_behavior",
        classification_status="classified",
    )
    axis_bundle = AxisSeriesBundle(
        trend_direction=_axis_result(sessions, "sideways"),
        trend_character=_axis_result(sessions, "trending"),
        volatility_state=_axis_result(sessions, "normal_vol"),
        breadth_state=_axis_result(sessions, "healthy_breadth"),
        event_calendar=_event_calendar(sessions),
        credit_funding_effective={day: classified_credit_stress for day in sessions},
        volume_liquidity_state={day: classified_liquidity_gap for day in sessions},
    )

    outputs = build_transition_risk_series(
        context=context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
        output_sessions=sessions[1:],
    )

    assert list(outputs) == sessions[1:]
    assert outputs[sessions[1]].state == "stable"
    assert outputs[sessions[1]].triggered_rules == ["state_confirmation_pending"]
    assert outputs[sessions[2]].state == "high_transition_risk"


def test_transition_risk_history_counts_axis_switches_per_session() -> None:
    sessions = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]
    history = build_transition_risk_history(
        sessions=sessions,
        trend_direction_stable_by_date={
            sessions[0]: "bull",
            sessions[1]: "bear",
            sessions[2]: "bear",
        },
        trend_character_stable_by_date={
            sessions[0]: "trending",
            sessions[1]: "transition",
            sessions[2]: "transition",
        },
        volatility_stable_by_date={
            sessions[0]: "normal_vol",
            sessions[1]: "normal_vol",
            sessions[2]: "high_vol",
        },
        breadth_stable_by_date={
            sessions[0]: "healthy_breadth",
            sessions[1]: "weak_breadth",
            sessions[2]: "weak_breadth",
        },
    )

    assert history.axis_switch_count_by_date == {
        sessions[0]: 0,
        sessions[1]: 3,
        sessions[2]: 1,
    }
