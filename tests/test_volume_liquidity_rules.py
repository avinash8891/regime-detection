"""TDD tests for v2 §1E Volume/Liquidity rule engine (Slice 2.7).

Spec references (docs/regime_engine_v2_spec.md):
    §1E lines 268-274 — `panic_volume`:
        volume_zscore_20d > 2.0 AND return_1d < -0.02
    §1E lines 276-280 — `liquidity_gap_behavior` (DEFERRED — see Ambiguity Log #40):
        gap_frequency_20d percentile_252d > 0.75
        AND intraday_range_percentile_252d > 0.75
    §1E line 282 — `normal_volume`: otherwise
    §1E lines 288-294 — risk_rank table (verbatim).

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic SPY-like inputs,
no toy a/b/c names, use the real production Pydantic config.
"""
from __future__ import annotations

import math

import pytest

from regime_detection.config import VolumeLiquidityRulesConfig
from regime_detection.volume_liquidity_rules import (
    VOLUME_LIQUIDITY_RISK_RANK,
    VolumeLiquidityRuleInputs,
    evaluate_liquidity_gap_behavior,
    evaluate_normal_volume,
    evaluate_panic_volume,
    evaluate_rules,
)


# v2 §1E lines 272-273, 278-279 — exact spec thresholds.
_SPEC_PANIC_ZSCORE_THRESHOLD = 2.0
_SPEC_PANIC_RETURN_THRESHOLD = -0.02
_SPEC_LIQGAP_FREQ_PCTL_THRESHOLD = 0.75
_SPEC_LIQGAP_RANGE_PCTL_THRESHOLD = 0.75


@pytest.fixture
def volume_liquidity_rules() -> VolumeLiquidityRulesConfig:
    return VolumeLiquidityRulesConfig(
        panic_volume_zscore_threshold=_SPEC_PANIC_ZSCORE_THRESHOLD,
        panic_volume_return_threshold=_SPEC_PANIC_RETURN_THRESHOLD,
        liquidity_gap_frequency_percentile_threshold=_SPEC_LIQGAP_FREQ_PCTL_THRESHOLD,
        liquidity_gap_intraday_range_percentile_threshold=_SPEC_LIQGAP_RANGE_PCTL_THRESHOLD,
    )


def _inputs(
    *,
    volume_zscore_20d: float = 0.0,
    return_1d: float = 0.0,
    gap_frequency_percentile_252d: float = 0.0,
    intraday_range_percentile_252d: float = 0.0,
) -> VolumeLiquidityRuleInputs:
    return VolumeLiquidityRuleInputs(
        volume_zscore_20d=volume_zscore_20d,
        return_1d=return_1d,
        gap_frequency_percentile_252d=gap_frequency_percentile_252d,
        intraday_range_percentile_252d=intraday_range_percentile_252d,
    )


# ---------- v2 §1E risk-rank table verbatim ---------------------------------


def test_volume_liquidity_risk_rank_matches_v2_spec_1e():
    # v2 §1E lines 288-294 verbatim.
    assert VOLUME_LIQUIDITY_RISK_RANK == {
        "normal_volume": 0,
        "unknown": 1,
        "liquidity_gap_behavior": 2,
        "panic_volume": 3,
    }


# ---------- panic_volume boundary tests (v2 §1E lines 270-274) ---------------


def test_panic_volume_fires_when_both_limbs_strictly_satisfied(volume_liquidity_rules):
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.01, return_1d=-0.025),
        volume_liquidity_rules,
    ) is True


def test_panic_volume_false_when_zscore_exactly_at_threshold(volume_liquidity_rules):
    # Strict `>` per spec line 272 — zscore == 2.0 does NOT satisfy.
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.0, return_1d=-0.025),
        volume_liquidity_rules,
    ) is False


def test_panic_volume_false_when_return_exactly_at_threshold(volume_liquidity_rules):
    # Strict `<` per spec line 273 — return_1d == -0.02 does NOT satisfy.
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.5, return_1d=-0.02),
        volume_liquidity_rules,
    ) is False


def test_panic_volume_false_when_return_above_threshold(volume_liquidity_rules):
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.5, return_1d=-0.019),
        volume_liquidity_rules,
    ) is False


def test_panic_volume_true_when_both_limbs_clearly_satisfied(volume_liquidity_rules):
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.5, return_1d=-0.025),
        volume_liquidity_rules,
    ) is True


def test_panic_volume_false_when_zscore_is_nan(volume_liquidity_rules):
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=float("nan"), return_1d=-0.025),
        volume_liquidity_rules,
    ) is False


def test_panic_volume_false_when_return_is_nan(volume_liquidity_rules):
    assert evaluate_panic_volume(
        _inputs(volume_zscore_20d=2.5, return_1d=float("nan")),
        volume_liquidity_rules,
    ) is False


# ---------- liquidity_gap_behavior: live predicate (Log #40 closure)


def test_liquidity_gap_behavior_fires_when_both_percentiles_above_threshold(
    volume_liquidity_rules,
):
    """v2 §1E lines 276-280: rule fires when BOTH the 252d percentile of
    `gap_frequency_20d` AND `intraday_range_percentile_252d` strictly
    exceed 0.75. Log #40 closure: the missing percentile input now ships
    from volatility_state_v2.gap_frequency_percentile_252d in the same
    commit that flipped this predicate from short-circuit-False."""
    inputs = _inputs(
        gap_frequency_percentile_252d=0.95,
        intraday_range_percentile_252d=0.95,
    )
    assert evaluate_liquidity_gap_behavior(inputs, volume_liquidity_rules) is True


def test_liquidity_gap_behavior_false_when_gap_freq_pct_at_or_below_threshold(
    volume_liquidity_rules,
):
    """Both inequalities are strict per spec — percentile EXACTLY at 0.75
    falsifies."""
    inputs = _inputs(
        gap_frequency_percentile_252d=0.75,
        intraday_range_percentile_252d=0.95,
    )
    assert evaluate_liquidity_gap_behavior(inputs, volume_liquidity_rules) is False


def test_liquidity_gap_behavior_false_when_intraday_pct_at_or_below_threshold(
    volume_liquidity_rules,
):
    inputs = _inputs(
        gap_frequency_percentile_252d=0.95,
        intraday_range_percentile_252d=0.75,
    )
    assert evaluate_liquidity_gap_behavior(inputs, volume_liquidity_rules) is False


def test_liquidity_gap_behavior_returns_false_on_nan_inputs(volume_liquidity_rules):
    """V1 §2.7 cold-start: NaN in either percentile input falsifies."""
    inputs = _inputs(
        gap_frequency_percentile_252d=float("nan"),
        intraday_range_percentile_252d=0.95,
    )
    assert evaluate_liquidity_gap_behavior(inputs, volume_liquidity_rules) is False
    inputs = _inputs(
        gap_frequency_percentile_252d=0.95,
        intraday_range_percentile_252d=float("nan"),
    )
    assert evaluate_liquidity_gap_behavior(inputs, volume_liquidity_rules) is False


# ---------- normal_volume = !panic AND !liquidity_gap (§1E line 282) ---------


def test_normal_volume_true_when_neither_panic_nor_liquidity_gap(volume_liquidity_rules):
    inputs = _inputs(volume_zscore_20d=0.5, return_1d=0.01)
    assert evaluate_normal_volume(inputs, volume_liquidity_rules) is True


def test_normal_volume_false_when_panic_fires(volume_liquidity_rules):
    inputs = _inputs(volume_zscore_20d=2.5, return_1d=-0.025)
    assert evaluate_normal_volume(inputs, volume_liquidity_rules) is False


def test_normal_volume_false_when_required_input_is_nan(volume_liquidity_rules):
    """If a required input is NaN we cannot assert "not panic AND not gap" so
    `normal_volume` returns False — the classifier maps that to `unknown`."""
    inputs = _inputs(volume_zscore_20d=float("nan"), return_1d=-0.025)
    assert evaluate_normal_volume(inputs, volume_liquidity_rules) is False


# ---------- Precedence walker (panic > liquidity_gap(deferred) > normal > unknown)


def test_evaluate_rules_returns_panic_volume_when_panic_fires(volume_liquidity_rules):
    label = evaluate_rules(
        inputs=_inputs(volume_zscore_20d=2.5, return_1d=-0.025),
        config=volume_liquidity_rules,
    )
    assert label == "panic_volume"


def test_evaluate_rules_returns_normal_volume_when_no_predicate_fires(
    volume_liquidity_rules,
):
    label = evaluate_rules(
        inputs=_inputs(volume_zscore_20d=0.4, return_1d=0.001),
        config=volume_liquidity_rules,
    )
    assert label == "normal_volume"


def test_evaluate_rules_returns_unknown_when_required_inputs_nan(
    volume_liquidity_rules,
):
    """Cold-start cannot conclude either rule — falls through to unknown."""
    label = evaluate_rules(
        inputs=_inputs(volume_zscore_20d=float("nan"), return_1d=-0.025),
        config=volume_liquidity_rules,
    )
    assert label == "unknown"


def test_evaluate_rules_returns_liquidity_gap_behavior_when_inputs_fire(
    volume_liquidity_rules,
):
    """Log #40 closure: when both percentile inputs strictly exceed 0.75
    and the panic rule does not fire, precedence walker returns
    `liquidity_gap_behavior` (the slot above `normal_volume` per §1E
    line 282 precedence)."""
    label = evaluate_rules(
        inputs=_inputs(
            volume_zscore_20d=0.5,
            return_1d=0.001,
            gap_frequency_percentile_252d=0.95,
            intraday_range_percentile_252d=0.95,
        ),
        config=volume_liquidity_rules,
    )
    assert label == "liquidity_gap_behavior"


def test_evaluate_rules_falls_through_to_normal_when_percentiles_below_threshold(
    volume_liquidity_rules,
):
    """When the percentile inputs do NOT exceed the 0.75 thresholds, the
    walker correctly falls through to normal_volume."""
    label = evaluate_rules(
        inputs=_inputs(
            volume_zscore_20d=0.5,
            return_1d=0.001,
            gap_frequency_percentile_252d=0.50,
            intraday_range_percentile_252d=0.50,
        ),
        config=volume_liquidity_rules,
    )
    assert label == "normal_volume"


# ---------- Config validation -----------------------------------------------


def test_rules_config_rejects_non_positive_zscore_threshold():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VolumeLiquidityRulesConfig(
            panic_volume_zscore_threshold=0.0,
            panic_volume_return_threshold=_SPEC_PANIC_RETURN_THRESHOLD,
            liquidity_gap_frequency_percentile_threshold=_SPEC_LIQGAP_FREQ_PCTL_THRESHOLD,
            liquidity_gap_intraday_range_percentile_threshold=_SPEC_LIQGAP_RANGE_PCTL_THRESHOLD,
        )


def test_rules_config_rejects_non_negative_return_threshold():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VolumeLiquidityRulesConfig(
            panic_volume_zscore_threshold=_SPEC_PANIC_ZSCORE_THRESHOLD,
            panic_volume_return_threshold=0.0,
            liquidity_gap_frequency_percentile_threshold=_SPEC_LIQGAP_FREQ_PCTL_THRESHOLD,
            liquidity_gap_intraday_range_percentile_threshold=_SPEC_LIQGAP_RANGE_PCTL_THRESHOLD,
        )


def test_rules_config_rejects_percentile_outside_unit_interval():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VolumeLiquidityRulesConfig(
            panic_volume_zscore_threshold=_SPEC_PANIC_ZSCORE_THRESHOLD,
            panic_volume_return_threshold=_SPEC_PANIC_RETURN_THRESHOLD,
            liquidity_gap_frequency_percentile_threshold=1.5,
            liquidity_gap_intraday_range_percentile_threshold=_SPEC_LIQGAP_RANGE_PCTL_THRESHOLD,
        )


# ---------- Default yaml sanity (slice-2.7 wiring) ---------------------------


def test_default_yaml_loads_volume_liquidity_config_with_spec_defaults():
    from regime_detection.config import load_default_regime_config

    cfg = load_default_regime_config()
    assert cfg.volume_liquidity_state is not None
    rules = cfg.volume_liquidity_state.rules
    assert math.isclose(rules.panic_volume_zscore_threshold, _SPEC_PANIC_ZSCORE_THRESHOLD)
    assert math.isclose(rules.panic_volume_return_threshold, _SPEC_PANIC_RETURN_THRESHOLD)
    assert math.isclose(
        rules.liquidity_gap_frequency_percentile_threshold,
        _SPEC_LIQGAP_FREQ_PCTL_THRESHOLD,
    )
    assert math.isclose(
        rules.liquidity_gap_intraday_range_percentile_threshold,
        _SPEC_LIQGAP_RANGE_PCTL_THRESHOLD,
    )
    # Ambiguity Log #41 — pinned hysteresis days.
    deesc = cfg.volume_liquidity_state.deescalation_days_by_label
    assert deesc["panic_volume"] == 3
    assert deesc["normal_volume"] == 0
    assert deesc["unknown"] == 2
