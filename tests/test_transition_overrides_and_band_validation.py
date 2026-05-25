"""Tests for the override-threshold config and band-monotonicity validator
added when the inline magic numbers in ``transition_risk_series`` were lifted
into ``TransitionScoreConfig.overrides``.

All tests use production component names (``credit_stress``,
``correlation_fragility``, ``macro_event``, ``trend_break``) and the real
``TransitionScoreConfig`` / ``TransitionOverrideThresholds`` classes loaded
from the shipping default config.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest
from pydantic import ValidationError

import pandas as pd

from regime_detection._config_evidence_strategy import (
    TransitionComponentScales,
    TransitionOverrideThresholds,
    TransitionScoreConfig,
)
from regime_detection.config import load_default_regime_config
from regime_detection.models import (
    DataQuality,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
)
from regime_detection.transition_risk import (
    TransitionRuleFlags,
    compose_transition_risk_output,
)
from regime_detection.transition_risk_series import (
    TransitionRiskHistory,
    TransitionScoreInputs,
    _apply_transition_state_debounce,
    _optional_float,
    build_transition_risk_outputs_by_date,
)
from regime_detection.transition_score import (
    ComposedTransitionScore,
    compose_transition_score_for_session,
)

pytestmark = pytest.mark.unit

logger = logging.getLogger(__name__)


def _default_transition_score_config() -> TransitionScoreConfig:
    config = load_default_regime_config()
    assert config.transition_score is not None
    return config.transition_score


def _bare_flags(**overrides: object) -> TransitionRuleFlags:
    kwargs: dict[str, object] = {
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


# --- band-monotonicity validator ---------------------------------------------


def test_default_config_passes_band_monotonicity_validator() -> None:
    # The shipping config must satisfy the new validator; otherwise we would
    # break every existing run.
    config = _default_transition_score_config()
    assert set(config.bands) >= {"stable", "weakening", "transition_warning", "high"}


def test_band_validator_rejects_non_monotonic_lower_bounds() -> None:
    base = _default_transition_score_config().model_dump()
    base["bands"]["weakening"] = (0.55, 0.75)  # overlap with transition_warning
    base["bands"]["transition_warning"] = (0.55, 0.75)
    with pytest.raises(ValidationError) as excinfo:
        TransitionScoreConfig(**base)
    assert "strictly increasing" in str(excinfo.value)


def test_band_validator_rejects_lo_ge_hi() -> None:
    base = _default_transition_score_config().model_dump()
    base["bands"]["weakening"] = (0.55, 0.55)
    with pytest.raises(ValidationError) as excinfo:
        TransitionScoreConfig(**base)
    assert "0.0 <= lo < hi <= 1.0" in str(excinfo.value)


def test_band_validator_rejects_out_of_unit_interval() -> None:
    base = _default_transition_score_config().model_dump()
    base["bands"]["high"] = (0.75, 1.5)
    with pytest.raises(ValidationError) as excinfo:
        TransitionScoreConfig(**base)
    assert "0.0 <= lo < hi <= 1.0" in str(excinfo.value)


def test_band_validator_requires_known_band_names() -> None:
    base = _default_transition_score_config().model_dump()
    del base["bands"]["weakening"]
    with pytest.raises(ValidationError) as excinfo:
        TransitionScoreConfig(**base)
    assert "missing required entries" in str(excinfo.value)


# --- primary_driver_min wiring ----------------------------------------------


def test_primary_drivers_threshold_uses_config_value() -> None:
    # Use real production component names from transition_score.compose_transition_score_for_session.
    score = ComposedTransitionScore(
        score=0.40,
        interpretation="weakening",
        components={
            "trend_break": 0.50,
            "credit_stress": 0.36,
            "macro_event": 0.20,
        },
    )

    # Default threshold (0.35) admits both trend_break and credit_stress.
    output_default = compose_transition_risk_output(score=score, flags=_bare_flags())
    assert output_default.primary_drivers == ["trend_break", "credit_stress"]

    # A stricter threshold passed by the caller (matching what
    # transition_risk_series threads from TransitionScoreConfig.overrides)
    # excludes credit_stress.
    output_strict = compose_transition_risk_output(
        score=score, flags=_bare_flags(), primary_driver_min=0.45
    )
    assert output_strict.primary_drivers == ["trend_break"]


# --- override threshold wiring through the per-session builder ---------------


def _single_session_inputs(
    *,
    credit_stress_component: float,
) -> tuple[date, dict[date, TransitionScoreInputs]]:
    # Pick numeric inputs whose component scores we can predict from
    # transition_score.py's normalization rules:
    #   credit_stress is set directly from the credit_funding_label table
    #   ("credit_stress" -> 0.75, "deleveraging" -> 1.0, "credit_calm" -> 0.0).
    day = date(2024, 6, 3)
    if credit_stress_component >= 0.75:
        credit_label = "credit_stress"
    else:
        credit_label = "credit_calm"
    inputs = TransitionScoreInputs(
        realized_vol_short=0.10,
        realized_vol_long=0.10,
        pct_above_50dma=0.80,
        avg_pairwise_corr_percentile_504d=0.10,
        drawdown_252d=0.0,
        event_calendar_labels=("normal_calendar",),
        spy_close=420.0,
        spy_sma_50=400.0,
        largest_eigenvalue_share_percentile_504d=0.10,
        effective_rank_percentile_504d=0.90,
        absorption_ratio_top3=0.30,
        credit_funding_label=credit_label,
        volume_liquidity_label="normal_volume",
        volume_zscore_20d=0.0,
        gap_frequency_percentile_252d=0.10,
        intraday_range_percentile_252d=0.10,
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
        cluster_id_now=1,
        cluster_id_5d_ago=1,
    )
    return day, {day: inputs}


def _trivial_history(day: date) -> TransitionRiskHistory:
    return TransitionRiskHistory(
        stable_changed_by_date={day: False},
        days_since_axis_switch_by_date={day: None},
        axis_switch_count_by_date={day: 0},
        recent_axis_switch_count_by_date={day: 0},
        prior_bear_by_date={day: False},
    )


def test_fragile_bull_override_threshold_is_config_driven() -> None:
    # Bull trend + elevated credit_stress (0.75 from the credit_funding label
    # mapping) should trigger fragile_bull when overrides.credit_stress <= 0.75
    # and should NOT trigger when overrides.credit_stress is raised above it.
    base_config = _default_transition_score_config()
    day, inputs_by_date = _single_session_inputs(credit_stress_component=0.75)
    history = _trivial_history(day)

    def _build(config: TransitionScoreConfig) -> str:
        outputs = build_transition_risk_outputs_by_date(
            sessions=[day],
            trend_direction_active_by_date={day: "bull"},
            trend_character_active_by_date={day: "uptrend"},
            volatility_state_active_by_date={day: "normal_vol"},
            breadth_state_active_by_date={day: "healthy_breadth"},
            close_by_date={day: 420.0},
            sma_50_by_date={day: 400.0},
            history=history,
            transition_score_inputs_by_date=inputs_by_date,
            transition_score_config=config,
        )
        return outputs[day].state

    # Default credit_stress=0.70 → triggered.
    assert _build(base_config) == "fragile_bull"

    # Raise threshold above the actual component value → no trigger; the
    # output falls back to the score-derived state.
    raised = base_config.model_copy(
        update={
            "overrides": TransitionOverrideThresholds(
                credit_stress=0.95,
                correlation_fragility=base_config.overrides.correlation_fragility,
                macro_event_min=base_config.overrides.macro_event_min,
                score_elevated_min=base_config.overrides.score_elevated_min,
                primary_driver_min=base_config.overrides.primary_driver_min,
            )
        }
    )
    assert _build(raised) != "fragile_bull"


def test_overrides_defaults_match_historical_inline_literals() -> None:
    # Guard against accidental default changes — the historical inline
    # literals were 0.70 / 0.70 / 1.0 / 0.35 / 0.35.
    defaults = TransitionOverrideThresholds()
    assert defaults.credit_stress == 0.70
    assert defaults.correlation_fragility == 0.70
    assert defaults.macro_event_min == 1.0
    assert defaults.score_elevated_min == 0.35
    assert defaults.primary_driver_min == 0.35


def test_default_config_omits_overrides_block_and_uses_defaults() -> None:
    # The shipping YAML does not declare an `overrides:` block. The config
    # must still load and expose the historical defaults.
    config = _default_transition_score_config()
    assert config.overrides == TransitionOverrideThresholds()


# --- Code 10: _optional_float must accept pd.NA without raising -------------


def test_optional_float_returns_none_for_pd_na() -> None:
    # Regression: float(pd.NA) raises TypeError; the helper must short-circuit
    # via pd.isna BEFORE attempting the cast.
    assert _optional_float(pd.NA) is None
    assert _optional_float(float("nan")) is None
    assert _optional_float(None) is None
    assert _optional_float(1.5) == 1.5


# --- Code 2: TransitionComponentScales preserves byte-identical scoring ------


def test_default_scales_match_historical_inline_literals() -> None:
    scales = TransitionComponentScales()
    assert scales.vol_acc_full_stress_ratio == 0.5
    assert scales.breadth_zero_stress_pct == 0.50
    assert scales.breadth_full_stress_range == 0.30
    assert scales.drawdown_full_stress == 0.15
    assert scales.ma_break_full_stress == 0.05
    assert scales.absorption_floor == 0.70
    assert scales.absorption_range == 0.25
    assert scales.volume_zscore_floor == 1.0
    assert scales.volume_zscore_range == 2.0


def test_default_config_omits_scales_block_and_uses_defaults() -> None:
    config = _default_transition_score_config()
    assert config.scales == TransitionComponentScales()


def test_compose_score_is_byte_identical_under_default_scales() -> None:
    # Pinned numeric outputs reproduced under the default scales. These were
    # computed by hand from the §4.2 formulas with the historical literals,
    # then confirmed by running compose_transition_score_for_session.
    config = _default_transition_score_config()
    composed = compose_transition_score_for_session(
        realized_vol_short=0.30,
        realized_vol_long=0.20,
        pct_above_50dma=0.30,
        avg_pairwise_corr_percentile_504d=0.80,
        drawdown_252d=-0.10,
        event_calendar_labels=("fed_week",),
        spy_close=400.0,
        spy_sma_50=420.0,
        absorption_ratio_top3=0.90,
        volume_zscore_20d=2.0,
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
        cluster_id_now=1,
        cluster_id_5d_ago=1,
        config=config,
    )
    assert composed.score is not None
    components = composed.components or {}
    assert components["volatility_acceleration"] == pytest.approx(1.0)
    assert components["breadth_deterioration"] == pytest.approx(0.6666666666666667)
    assert components["trend_break"] == pytest.approx(0.9523809523809523)
    assert components["correlation_fragility"] == pytest.approx(0.80)
    assert components["liquidity_stress"] == pytest.approx(0.5)


def test_scales_are_tunable_via_config() -> None:
    # Tightening drawdown_full_stress halves the input needed for full
    # trend_drawdown saturation, so a smaller drawdown now saturates.
    base = _default_transition_score_config()
    tighter = base.model_copy(
        update={"scales": TransitionComponentScales(drawdown_full_stress=0.05)}
    )

    # 5% drawdown under tighter scales saturates trend_drawdown.
    # Provide credit_funding_label + volume_liquidity_label so the
    # configured weight coverage clears minimum_component_weight_coverage
    # (default 0.75) and the score is computed instead of returning None.
    kwargs = dict(
        realized_vol_short=0.20,
        realized_vol_long=0.20,
        pct_above_50dma=0.80,
        avg_pairwise_corr_percentile_504d=0.10,
        drawdown_252d=-0.05,
        event_calendar_labels=("normal_calendar",),
        credit_funding_label="credit_calm",
        volume_liquidity_label="normal_volume",
        hmm_top_state_prob_now=0.50,
        hmm_top_state_prob_5d_ago=0.50,
        change_point_score=0.0,
        cluster_id_now=1,
        cluster_id_5d_ago=1,
    )
    composed = compose_transition_score_for_session(config=tighter, **kwargs)
    components = composed.components or {}
    assert components["trend_break"] == pytest.approx(1.0)

    # Same input under default scales is well below saturation
    # (5%/15% = 0.333…).
    composed_default = compose_transition_score_for_session(config=base, **kwargs)
    default_components = composed_default.components or {}
    assert default_components["trend_break"] == pytest.approx(0.3333333333333333)


# --- Code 6: initial_active_state opt-in seed --------------------------------


def _raw(state: str) -> TransitionRiskOutput:
    return TransitionRiskOutput(
        state=state,
        score=None,
        score_components=None,
        primary_drivers=[],
        triggered_rules=[],
        evidence=TransitionRiskEvidencePayload(
            triggered_rules=[],
            stable_changed_today=False,
            days_since_axis_switch=None,
            axis_switch_count=0,
            recent_axis_switch_count=0,
        ),
        data_quality=DataQuality(status="ok"),
    )


def _confirmation_windows() -> dict[str, int]:
    return _default_transition_score_config().state_confirmation_days


def test_first_session_bypass_is_preserved_when_initial_active_state_unset() -> None:
    # Default behavior: the first session's raw state is accepted immediately
    # even when its confirmation window is > 1. This is the documented
    # historical behavior and golden fixtures depend on it.
    sessions = [date(2024, 1, 2), date(2024, 1, 3)]
    raw = {sessions[0]: _raw("weakening"), sessions[1]: _raw("weakening")}
    debounced = _apply_transition_state_debounce(
        sessions=sessions,
        raw_outputs=raw,
        state_confirmation_days=_confirmation_windows(),
    )
    assert debounced[sessions[0]].state == "weakening"
    assert debounced[sessions[0]].triggered_rules == []


def test_initial_active_state_seed_forces_first_session_confirmation() -> None:
    # When seeded with 'stable', a first-session 'weakening' raw print must
    # clear the 2-print confirmation window before becoming public. The
    # first session reports the seeded state with state_confirmation_pending.
    sessions = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    raw = {
        sessions[0]: _raw("weakening"),
        sessions[1]: _raw("weakening"),
        sessions[2]: _raw("stable"),
    }
    debounced = _apply_transition_state_debounce(
        sessions=sessions,
        raw_outputs=raw,
        state_confirmation_days=_confirmation_windows(),
        initial_active_state="stable",
    )
    assert debounced[sessions[0]].state == "stable"
    assert "state_confirmation_pending" in debounced[sessions[0]].triggered_rules
    # Second 'weakening' confirms — promotes on day 2.
    assert debounced[sessions[1]].state == "weakening"
    # Day 3 stable confirms in 1 print.
    assert debounced[sessions[2]].state == "stable"


def test_initial_active_state_invalid_raises() -> None:
    sessions = [date(2024, 1, 2)]
    raw = {sessions[0]: _raw("stable")}
    with pytest.raises(ValueError, match="initial_active_state"):
        _apply_transition_state_debounce(
            sessions=sessions,
            raw_outputs=raw,
            state_confirmation_days=_confirmation_windows(),
            initial_active_state="not_a_real_state",
        )


def test_config_initial_active_state_unknown_state_rejected() -> None:
    base = _default_transition_score_config().model_dump()
    base["initial_active_state"] = "totally_made_up"
    with pytest.raises(ValidationError, match="initial_active_state"):
        TransitionScoreConfig(**base)


def test_default_config_initial_active_state_is_none() -> None:
    # The shipping YAML does not declare initial_active_state; default must
    # remain None so backfill behavior is unchanged.
    assert _default_transition_score_config().initial_active_state is None
