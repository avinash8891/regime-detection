from __future__ import annotations

from datetime import date
from typing import get_type_hints

import pandas as pd
import pytest
from pydantic import ValidationError

from regime_detection.config import load_default_regime_config
from regime_detection.models import EventCalendarOutput, TransitionRiskOutput
from regime_detection.transition_risk import (
    build_transition_risk_output_from_flags,
    classify_transition_risk,
)
from regime_detection.transition_risk_series import (
    EventCalendarLabel,
    TransitionRiskHistory,
    TransitionScoreInputs,
    _build_transition_score_inputs_by_date,
    build_transition_risk_outputs_by_date,
)
from regime_detection.transition_score import compose_transition_score_for_session


_AS_OF = date(2024, 1, 2)


def _classify(**overrides):
    kwargs = {
        "as_of_date": _AS_OF,
        "trend_direction_active": "bull",
        "prior_bear_in_last_60_sessions": False,
        "trend_character_active": "trending",
        "volatility_state_active": "normal_vol",
        "breadth_state_active": "healthy_breadth",
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "close": 100.0,
        "sma_50": 95.0,
    }
    kwargs.update(overrides)
    return classify_transition_risk(**kwargs)


@pytest.mark.parametrize(
    ("flags", "expected_label", "expected_warnings"),
    [
        (
            {
                "crisis_override": True,
                "bear_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "crisis_override",
            [
                "crisis_override",
                "bear_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": True,
                "sideways_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "bear_stress_warning",
            [
                "bear_stress_warning",
                "sideways_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
                "post_switch_cooldown",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "sideways_stress_warning": True,
                "bull_fragile_warning": True,
                "recovery_attempt": True,
                "post_switch_cooldown": True,
                "any_unknown": True,
            },
            "sideways_stress_warning",
            [
                "sideways_stress_warning",
                "bull_fragile_warning",
                "recovery_attempt",
                "post_switch_cooldown",
            ],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "bull_fragile_warning": False,
                "recovery_attempt": False,
                "post_switch_cooldown": False,
                "any_unknown": True,
            },
            "unknown",
            [],
        ),
        (
            {
                "crisis_override": False,
                "bear_stress_warning": False,
                "bull_fragile_warning": False,
                "recovery_attempt": False,
                "post_switch_cooldown": False,
                "any_unknown": False,
            },
            "stable",
            [],
        ),
    ],
)
def test_build_transition_risk_output_from_flags_applies_precedence(
    flags,
    expected_label,
    expected_warnings,
) -> None:
    output = build_transition_risk_output_from_flags(
        stable_changed_today=True,
        days_since_axis_switch=0,
        **flags,
    )

    assert output.label == expected_label
    assert output.evidence["warnings_active"] == expected_warnings
    assert output.evidence["stable_changed_today"] is True
    assert output.evidence["days_since_axis_switch"] == 0


def test_classify_transition_risk_bear_stress_warning_predicate() -> None:
    output = _classify(
        trend_direction_active="bear",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
    )

    assert output.label == "bear_stress_warning"
    assert output.evidence["warnings_active"] == ["bear_stress_warning"]


def test_classify_transition_risk_v2_sideways_warning_is_gated() -> None:
    v1_output = _classify(
        trend_direction_active="sideways",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
        allow_v2_warnings=False,
    )
    v2_output = _classify(
        trend_direction_active="sideways",
        volatility_state_active="high_vol",
        breadth_state_active="weak_breadth",
        allow_v2_warnings=True,
    )

    assert v1_output.label == "stable"
    assert v2_output.label == "sideways_stress_warning"


def test_classify_transition_risk_recovery_attempt_from_prior_bear_price_and_breadth() -> None:
    output = _classify(
        prior_bear_in_last_60_sessions=True,
        trend_character_active="transition",
        close=101.0,
        sma_50=100.0,
        breadth_state_active="recovery_breadth",
    )

    assert output.label == "recovery_attempt"
    assert output.evidence["warnings_active"] == ["recovery_attempt"]


def test_classify_transition_risk_post_switch_cooldown_is_inclusive_through_day_five() -> None:
    day_five = _classify(days_since_axis_switch=5, stable_changed_today=False)
    day_six = _classify(days_since_axis_switch=6, stable_changed_today=False)

    assert day_five.label == "post_switch_cooldown"
    assert day_six.label == "stable"


def test_classify_transition_risk_crisis_override_suppresses_cooldown_warning() -> None:
    output = _classify(
        volatility_state_active="crisis_vol",
        days_since_axis_switch=0,
        stable_changed_today=True,
    )

    assert output.label == "crisis_override"
    assert output.evidence["warnings_active"] == ["crisis_override"]


def test_transition_risk_evidence_preserves_legacy_dict_access_and_dump_shape() -> None:
    output = TransitionRiskOutput(
        label="stable",
        evidence={
            "warnings_active": [],
            "stable_changed_today": False,
            "days_since_axis_switch": None,
        },
    )

    expected = {
        "warnings_active": [],
        "stable_changed_today": False,
        "days_since_axis_switch": None,
    }
    assert output.evidence["warnings_active"] == []
    assert output.evidence.get("stable_changed_today") is False
    assert output.evidence.get("missing", "fallback") == "fallback"
    assert output.evidence == expected
    assert output.evidence.model_dump() == expected
    assert output.evidence.model_dump_json() == (
        '{"warnings_active":[],"stable_changed_today":false,'
        '"days_since_axis_switch":null}'
    )
    assert output.model_dump()["evidence"] == expected


@pytest.mark.parametrize("bad_key", ["warning_active", "reason"])
def test_transition_risk_evidence_rejects_unknown_keys(bad_key: str) -> None:
    with pytest.raises(ValidationError, match=bad_key):
        TransitionRiskOutput(
            label="stable",
            evidence={
                "warnings_active": [],
                "stable_changed_today": False,
                "days_since_axis_switch": None,
                bad_key: "unexpected",
            },
        )


def test_build_transition_score_inputs_returns_typed_optional_hmm_and_change_point_values() -> None:
    sessions = [date(2024, 1, day) for day in range(2, 10)]
    index = pd.DatetimeIndex(sessions)

    inputs_by_date = _build_transition_score_inputs_by_date(
        sessions=sessions,
        realized_vol_short=pd.Series([12.0] * len(sessions), index=index),
        realized_vol_long=pd.Series([10.0] * len(sessions), index=index),
        pct_above_50dma=pd.Series([0.45] * len(sessions), index=index),
        avg_pairwise_corr_percentile_504d=pd.Series([0.60] * len(sessions), index=index),
        drawdown_252d=pd.Series([-0.10] * len(sessions), index=index),
        event_calendar={
            day: EventCalendarOutput(
                raw_label="normal",
                stable_label="normal",
                active_label="normal",
                evidence={"upcoming_events": []},
            )
            for day in sessions
        },
        hmm_top_state_prob=pd.Series(
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.70, 0.75],
            index=index,
        ),
        change_point_score=pd.Series([0.20] * len(sessions), index=index),
    )

    first_day = sessions[0]
    shifted_day = sessions[5]
    assert isinstance(inputs_by_date[first_day], TransitionScoreInputs)
    assert inputs_by_date[first_day].hmm_top_state_prob_now == pytest.approx(0.10)
    assert pd.isna(inputs_by_date[first_day].hmm_top_state_prob_5d_ago)
    assert inputs_by_date[first_day].change_point_score == pytest.approx(0.20)
    assert inputs_by_date[shifted_day].hmm_top_state_prob_now == pytest.approx(0.65)
    assert inputs_by_date[shifted_day].hmm_top_state_prob_5d_ago == pytest.approx(0.10)


def test_transition_risk_score_inputs_match_direct_composer_for_optional_hmm_and_change_point() -> None:
    cfg = load_default_regime_config().transition_score
    assert cfg is not None
    session = date(2024, 1, 9)
    score_inputs = TransitionScoreInputs(
        realized_vol_short=12.0,
        realized_vol_long=10.0,
        pct_above_50dma=0.45,
        avg_pairwise_corr_percentile_504d=0.60,
        drawdown_252d=-0.10,
        event_calendar_label="cpi_week",
        hmm_top_state_prob_now=0.70,
        hmm_top_state_prob_5d_ago=0.30,
        change_point_score=0.50,
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
        event_calendar_label=score_inputs.event_calendar_label,
        hmm_top_state_prob_now=score_inputs.hmm_top_state_prob_now,
        hmm_top_state_prob_5d_ago=score_inputs.hmm_top_state_prob_5d_ago,
        change_point_score=score_inputs.change_point_score,
        config=cfg,
    )

    assert outputs[session].score == expected.score
    assert outputs[session].score_interpretation == expected.interpretation
    assert outputs[session].score_components == expected.components


def test_transition_score_inputs_event_calendar_label_is_closed_type() -> None:
    assert get_type_hints(TransitionScoreInputs)["event_calendar_label"] == EventCalendarLabel
    with pytest.raises(ValueError, match="unknown event_calendar_label"):
        TransitionScoreInputs(
            realized_vol_short=12.0,
            realized_vol_long=10.0,
            pct_above_50dma=0.45,
            avg_pairwise_corr_percentile_504d=0.60,
            drawdown_252d=-0.10,
            event_calendar_label="vendor_changed_name",  # type: ignore[arg-type]
        )
