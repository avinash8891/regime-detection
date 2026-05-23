from __future__ import annotations

from datetime import date

import pytest

from regime_detection.comparison import (
    GateMetric,
    StrategyMetrics,
    compute_v1_v2_diff,
    evaluate_v2_gate,
)
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    MonetaryPressureOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    RegimeTimeline,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
)


# ---------- evaluate_v2_gate (v2 §9.1) --------------------------------------


def _baseline_metrics() -> StrategyMetrics:
    """Realistic V1 baseline anchored on SPY-like long-only returns
    around the V1 walkforward window (no toy values)."""
    return StrategyMetrics(
        max_drawdown=-0.18,
        sharpe=0.85,
        mean_crisis_detection_lag_days=4.0,
        false_switch_rate=0.12,
    )


def test_evaluate_v2_gate_passes_when_drawdown_is_less_negative() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=-0.12,  # less negative = better
        sharpe=v1.sharpe,
        mean_crisis_detection_lag_days=v1.mean_crisis_detection_lag_days,
        false_switch_rate=v1.false_switch_rate,
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is True
    assert result.winning_metrics == (GateMetric.LOWER_DRAWDOWN,)


def test_evaluate_v2_gate_passes_when_sharpe_higher() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=v1.max_drawdown,
        sharpe=1.05,  # higher = better
        mean_crisis_detection_lag_days=v1.mean_crisis_detection_lag_days,
        false_switch_rate=v1.false_switch_rate,
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is True
    assert result.winning_metrics == (GateMetric.HIGHER_SHARPE,)


def test_evaluate_v2_gate_passes_when_detection_lag_lower() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=v1.max_drawdown,
        sharpe=v1.sharpe,
        mean_crisis_detection_lag_days=2.5,  # 1.5 days earlier
        false_switch_rate=v1.false_switch_rate,
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is True
    assert result.winning_metrics == (GateMetric.EARLIER_CRISIS_DETECTION,)


def test_evaluate_v2_gate_passes_when_false_switch_rate_lower() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=v1.max_drawdown,
        sharpe=v1.sharpe,
        mean_crisis_detection_lag_days=v1.mean_crisis_detection_lag_days,
        false_switch_rate=0.08,
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is True
    assert result.winning_metrics == (GateMetric.LOWER_FALSE_SWITCH_RATE,)


def test_evaluate_v2_gate_collects_all_winners_not_just_first() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=-0.10,
        sharpe=1.10,
        mean_crisis_detection_lag_days=2.0,
        false_switch_rate=0.07,
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is True
    assert set(result.winning_metrics) == {
        GateMetric.LOWER_DRAWDOWN,
        GateMetric.HIGHER_SHARPE,
        GateMetric.EARLIER_CRISIS_DETECTION,
        GateMetric.LOWER_FALSE_SWITCH_RATE,
    }


def test_evaluate_v2_gate_fails_when_metrics_identical() -> None:
    v1 = _baseline_metrics()
    v2 = _baseline_metrics()

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is False
    assert result.winning_metrics == ()


def test_evaluate_v2_gate_fails_when_v2_worse_on_every_metric() -> None:
    v1 = _baseline_metrics()
    v2 = StrategyMetrics(
        max_drawdown=-0.25,  # more negative = worse
        sharpe=0.60,  # lower = worse
        mean_crisis_detection_lag_days=6.0,  # higher = worse
        false_switch_rate=0.20,  # higher = worse
    )

    result = evaluate_v2_gate(v1_metrics=v1, v2_metrics=v2)

    assert result.passed is False
    assert result.winning_metrics == ()


# ---------- compute_v1_v2_diff (v2 §9.3 A/B shadow review) -------------------


def _data_quality() -> DataQuality:
    return DataQuality(status="ok")


def _axis(label: str) -> AxisOutput:
    return AxisOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={},
        data_quality=_data_quality(),
    )


def _breadth(label: str) -> BreadthStateOutput:
    return BreadthStateOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={},
        data_quality=_data_quality(),
        mode="etf_proxy",
    )


def _output(as_of_date: date, *, trend_label: str = "bull") -> RegimeOutput:
    return RegimeOutput(
        engine_version="regime-engine-v-test",
        config_version="test",
        as_of_date=as_of_date,
        market="SPY",
        trend_direction=_axis(trend_label),
        trend_character=_axis("steady"),
        volatility_state=_axis("normal_vol"),
        breadth_state=_breadth("healthy"),
        structural_causal_state=StructuralCausalState(
            event_calendar=EventCalendarOutput(
                primary_label="normal_calendar",
                matching_labels=("normal_calendar",),
                evidence={},
            ),
            monetary_pressure=MonetaryPressureOutput(
                label="unknown",
                evidence={},
                data_quality=_data_quality(),
            ),
        ),
        network_fragility=NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={},
            data_quality=_data_quality(),
        ),
        transition_risk=TransitionRiskOutput(
            state="stable",
            evidence=TransitionRiskEvidencePayload(
                triggered_rules=[],
                stable_changed_today=False,
                days_since_axis_switch=None,
                axis_switch_count=0,
                recent_axis_switch_count=0,
            ),
            score=0.10,
            score_components={"trend_break": 0.10},
            data_quality=_data_quality(),
        ),
        strategy_response=StrategyResponse(
            position_size_multiplier=1.0,
            allow_trend_following=True,
            allow_mean_reversion=True,
            leverage_allowed=False,
            allow_buy_dip=True,
            allow_breakout=True,
            allow_shorts=False,
            require_confirmation_for_new_longs=False,
            require_confirmation_for_shorts=True,
            log_for_review=False,
            modifiers_applied=[],
        ),
    )


def _timeline(*outputs: RegimeOutput) -> RegimeTimeline:
    return RegimeTimeline(
        engine_version="regime-engine-v-test",
        config_version="test",
        market="SPY",
        start_date=outputs[0].as_of_date,
        end_date=outputs[-1].as_of_date,
        trading_calendar="XNYS",
        outputs=list(outputs),
    )


def _two_date_timeline() -> RegimeTimeline:
    return _timeline(
        _output(date(2023, 12, 13)),
        _output(date(2023, 12, 14)),
    )


def test_compute_v1_v2_diff_zero_disagreements_for_identical_timelines() -> None:
    timeline_a = _two_date_timeline()
    timeline_b = _two_date_timeline()

    diff = compute_v1_v2_diff(timeline_a, timeline_b)

    assert diff.label_diffs == ()
    assert diff.v1_only_top_level_fields == ()
    assert diff.v2_only_top_level_fields == ()


def test_compute_v1_v2_diff_raises_on_length_mismatch() -> None:
    long_tl = _two_date_timeline()
    short_tl = _timeline(_output(date(2023, 12, 14)))

    with pytest.raises(ValueError, match="length mismatch"):
        compute_v1_v2_diff(long_tl, short_tl)


def test_compute_v1_v2_diff_detects_axis_label_disagreement() -> None:
    """Force a label disagreement by mutating one timeline's first output."""
    timeline_a = _two_date_timeline()
    timeline_b = _two_date_timeline()

    # Override timeline_b.outputs[0].trend_direction.active_label by deep-copy
    # mutation via model_copy.
    mutated_outputs = list(timeline_b.outputs)
    bad = mutated_outputs[0].model_copy(deep=True)
    bad.trend_direction = bad.trend_direction.model_copy(update={"active_label": "bear"})
    mutated_outputs[0] = bad
    timeline_b_mut = timeline_b.model_copy(update={"outputs": mutated_outputs})

    diff = compute_v1_v2_diff(timeline_a, timeline_b_mut)

    matches = [d for d in diff.label_diffs if d.axis == "trend_direction"]
    assert len(matches) == 1
    assert matches[0].v1_active_label == timeline_a.outputs[0].trend_direction.active_label
    assert matches[0].v2_active_label == "bear"


def test_compute_v1_v2_diff_reports_granular_status_for_unknown_labels() -> None:
    timeline_a = _timeline(_output(date(2023, 12, 13)))
    timeline_b = _timeline(_output(date(2023, 12, 13)))

    left_outputs = list(timeline_a.outputs)
    left = left_outputs[0].model_copy(deep=True)
    left.network_fragility = left.network_fragility.model_copy(
        update={
            "active_label": "unknown",
            "raw_label": "unknown",
            "stable_label": "unknown",
            "classification_status": "no_rule_fired",
        }
    )
    left_outputs[0] = left

    right_outputs = list(timeline_b.outputs)
    right = right_outputs[0].model_copy(deep=True)
    right.network_fragility = right.network_fragility.model_copy(
        update={
            "active_label": "unknown",
            "raw_label": "unknown",
            "stable_label": "unknown",
            "classification_status": "stale_data",
        }
    )
    right_outputs[0] = right

    diff = compute_v1_v2_diff(
        timeline_a.model_copy(update={"outputs": left_outputs}),
        timeline_b.model_copy(update={"outputs": right_outputs}),
    )

    matches = [d for d in diff.label_diffs if d.axis == "network_fragility"]
    assert len(matches) == 1
    assert matches[0].v1_active_label == "no_rule_fired"
    assert matches[0].v2_active_label == "stale_data"
