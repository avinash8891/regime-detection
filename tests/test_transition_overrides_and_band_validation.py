"""Tests for the override-threshold config and band-monotonicity validator
added when the inline magic numbers in ``transition_risk_series`` were lifted
into ``TransitionScoreConfig.overrides``.

All tests use production component names (``credit_stress``,
``correlation_fragility``, ``macro_event``, ``trend_break``) and the real
``TransitionScoreConfig`` / ``TransitionOverrideThresholds`` classes loaded
from the shipping default config.
"""

from __future__ import annotations

import copy
import logging
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from regime_detection._config_evidence_strategy import (
    TransitionOverrideThresholds,
    TransitionScoreConfig,
)
from regime_detection.config import load_default_regime_config
from regime_detection.transition_risk import (
    TransitionRuleFlags,
    compose_transition_risk_output,
)
from regime_detection.transition_risk_series import (
    TransitionRiskHistory,
    TransitionScoreInputs,
    build_transition_risk_outputs_by_date,
)
from regime_detection.transition_score import ComposedTransitionScore

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
