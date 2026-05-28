"""RED-phase tests for V2 Slice 5.2 Strategy Family Constraints.

Pins V2 spec §5.2 (docs/regime_engine_v2_spec.md lines 2570-2652).

Real-config-only: tests construct ``StrategyFamilyConstraintsConfig`` via
``load_default_regime_config()`` (one home for the yaml round-trip) and use
the production family-name constants from
``regime_detection.strategy_family_constraints.STRATEGY_FAMILIES``.
"""

from __future__ import annotations

from datetime import date

import pytest

from regime_detection.config import (
    FamilyOverride,
    StrategyFamilyConstraintsConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.models import RegimeOutput, StrategyFamilyConstraint
from regime_detection.strategy_family_constraints import (
    STRATEGY_FAMILIES,
    resolve_strategy_family_constraints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_config() -> StrategyFamilyConstraintsConfig:
    cfg = load_default_regime_config().strategy_family_constraints
    assert isinstance(
        cfg, StrategyFamilyConstraintsConfig
    ), "default V2 config must ship strategy_family_constraints"
    return cfg


# ===========================================================================
# Group A — inheritance & overrides
# ===========================================================================


@pytest.mark.unit
def test_default_neutral_inheritance_when_active_cohort_is_default_neutral() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="default_neutral", config=_real_config()
    )
    assert out["trend_following"].allowed is True
    assert out["trend_following"].max_lookback_days == 50
    assert out["trend_following"].require_breadth_confirmation is False
    assert out["trend_following"].min_adx == 20
    assert out["mean_reversion"].allowed is True
    assert out["mean_reversion"].max_holding_days == 5
    assert out["mean_reversion"].require_volume_confirmation is True
    assert out["breakout"].allowed is False
    assert out["breakout"].reason == "false_breakout_rate_high_in_chop"
    assert out["short_vol"].allowed is False
    assert out["short_vol"].reason == "rising_fragility_or_crisis"
    assert out["long_vol"].allowed is True
    assert out["long_vol"].event_window_only is True


@pytest.mark.unit
def test_crisis_specialist_overrides_all_5_families() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="crisis_specialist", config=_real_config()
    )
    assert out["trend_following"].allowed is False
    assert out["trend_following"].reason == "false_signals_in_chop"
    assert out["mean_reversion"].allowed is False
    assert out["mean_reversion"].reason == "knife_catching"
    assert out["breakout"].allowed is False
    assert out["short_vol"].allowed is False
    assert out["long_vol"].allowed is True
    assert out["long_vol"].event_window_only is False  # override flips default True


@pytest.mark.unit
def test_euphoria_specialist_overrides_only_mean_reversion_others_inherit() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="euphoria_specialist", config=_real_config()
    )
    assert out["mean_reversion"].allowed is False
    assert out["mean_reversion"].reason == "do_not_fade_strength"
    # Inherited from default_neutral:
    assert out["trend_following"].allowed is True
    assert out["trend_following"].max_lookback_days == 50
    assert out["breakout"].allowed is False
    assert out["short_vol"].allowed is False
    assert out["long_vol"].event_window_only is True


@pytest.mark.unit
def test_easing_specialist_inherits_everything_no_overrides() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="easing_specialist", config=_real_config()
    )
    default = resolve_strategy_family_constraints(
        active_cohort="default_neutral", config=_real_config()
    )
    for family in STRATEGY_FAMILIES:
        assert out[family].model_dump() == default[family].model_dump()


@pytest.mark.unit
def test_bull_low_vol_specialist_overrides_max_lookback_days_to_200() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="bull_low_vol_specialist", config=_real_config()
    )
    assert out["trend_following"].allowed is True
    assert out["trend_following"].max_lookback_days == 200
    assert out["trend_following"].min_adx == 15


@pytest.mark.unit
def test_bull_low_vol_specialist_short_vol_max_position_pct_quarter() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="bull_low_vol_specialist", config=_real_config()
    )
    assert out["short_vol"].allowed is True
    assert out["short_vol"].max_position_pct == 0.25


@pytest.mark.unit
def test_allowed_specialist_override_clears_inherited_block_reason() -> None:
    cfg = _real_config()

    bull_low_vol = resolve_strategy_family_constraints(
        active_cohort="bull_low_vol_specialist", config=cfg
    )
    tightening = resolve_strategy_family_constraints(
        active_cohort="tightening_specialist", config=cfg
    )

    assert bull_low_vol["breakout"].allowed is True
    assert bull_low_vol["breakout"].reason is None
    assert bull_low_vol["short_vol"].allowed is True
    assert bull_low_vol["short_vol"].reason is None
    assert tightening["breakout"].allowed is True
    assert tightening["breakout"].reason is None


@pytest.mark.unit
def test_chop_mean_reversion_specialist_disables_trend_following_and_breakout() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="chop_mean_reversion_specialist", config=_real_config()
    )
    assert out["trend_following"].allowed is False
    assert out["breakout"].allowed is False
    assert out["mean_reversion"].allowed is True
    assert out["mean_reversion"].max_holding_days == 10
    assert out["mean_reversion"].require_volume_confirmation is False


@pytest.mark.unit
def test_recovery_specialist_disables_short_vol() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="recovery_specialist", config=_real_config()
    )
    assert out["short_vol"].allowed is False
    assert out["short_vol"].reason == "recovery_can_relapse"


@pytest.mark.unit
def test_unknown_cohort_falls_back_to_default_neutral_baseline() -> None:
    out = resolve_strategy_family_constraints(
        active_cohort="nonexistent_cohort", config=_real_config()
    )
    default = resolve_strategy_family_constraints(
        active_cohort="default_neutral", config=_real_config()
    )
    for family in STRATEGY_FAMILIES:
        assert out[family].model_dump() == default[family].model_dump()


@pytest.mark.unit
def test_resolve_uses_real_default_config() -> None:
    cfg = load_default_regime_config()
    out = resolve_strategy_family_constraints(
        active_cohort="default_neutral", config=cfg.strategy_family_constraints
    )
    assert set(STRATEGY_FAMILIES).issubset(set(out.keys()))


# ===========================================================================
# Group B — wire / model contracts
# ===========================================================================


@pytest.mark.unit
def test_strategy_family_constraint_omits_none_fields_in_json() -> None:
    sfc = StrategyFamilyConstraint(allowed=True)
    js = sfc.model_dump_json()
    assert "null" not in js
    assert js == '{"allowed":true}'


@pytest.mark.unit
def test_default_neutral_requires_allowed_field_for_every_family() -> None:
    bad = StrategyFamilyConstraintsConfig(
        default_neutral={
            "trend_following": FamilyOverride(max_lookback_days=50),  # allowed missing
        },
        overrides={},
    )
    with pytest.raises(ValueError, match="trend_following"):
        resolve_strategy_family_constraints(active_cohort="default_neutral", config=bad)


# ===========================================================================
# Group C — engine wire-in
# ===========================================================================


@pytest.mark.integration
def test_regime_output_carries_strategy_family_constraints_when_configured(
    classified_golden_outputs: dict[date, RegimeOutput],
) -> None:
    assert classified_golden_outputs
    for _as_of, out in classified_golden_outputs.items():
        assert out.strategy_family_constraints is not None
        assert set(STRATEGY_FAMILIES).issubset(
            set(out.strategy_family_constraints.keys())
        )
        for family in STRATEGY_FAMILIES:
            assert isinstance(
                out.strategy_family_constraints[family], StrategyFamilyConstraint
            )


@pytest.mark.integration
def test_regime_output_omits_strategy_family_constraints_when_config_absent(
    v2_market_df_for_asof,
    golden_rows: list[dict[str, object]],
    synthetic_v2_kwargs_for_market_data,
) -> None:
    engine = RegimeEngine()
    as_of = max(date.fromisoformat(str(row["as_of_date"])) for row in golden_rows)
    market_data = v2_market_df_for_asof(as_of)
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    no_sfc = kwargs["config"].model_copy(update={"strategy_family_constraints": None})
    assert no_sfc.strategy_family_constraints is None
    kwargs["config"] = no_sfc
    out = engine.classify(
        as_of_date=as_of,
        market_data=market_data,
        **kwargs,
    )
    assert out.strategy_family_constraints is None
    assert "strategy_family_constraints" not in out.model_dump()
