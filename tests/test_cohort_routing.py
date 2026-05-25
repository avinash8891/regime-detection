"""RED-phase tests for V2 Slice 5.1 Agent Cohort Routing.

Pins V2 spec §5.1 (docs/regime_engine_v2_spec.md lines 2478-2566) per
Ambiguity Log #53. These tests target the public surface:

    evaluate_cohort_routing(...)
    AgentRouting (Pydantic model)
    CohortRoutingConfig (Pydantic sub-config)
    RegimeConfig.cohort_routing (None by default → V1 wire byte-identity)
    RegimeOutput.agent_routing  (None by default → omitted from wire)

All tests use real V2 axis label strings drawn from the production
``Literal[...]`` types in ``regime_detection.models`` /
``regime_detection.trend_direction`` / ``regime_detection.volatility_state`` /
``regime_detection.breadth_state`` / ``regime_detection.network_fragility_rules``.
Zero toy strings.

Per the TDD RED-phase contract these MUST fail at collection time
(ImportError on ``cohort_routing`` module / ``AgentRouting`` /
``CohortRoutingConfig``) until Slice 5.1 GREEN ships.
"""

from __future__ import annotations

from datetime import date
import logging

import pytest
from pydantic import ValidationError

# Real V2 axis label types — imported here so the test file fails fast
# (NameError, not toy-string compile) if the production enums move.
from regime_detection.breadth_state import BreadthLabel  # noqa: F401
from regime_detection.config import (
    CohortRoutingConfig,
    load_default_regime_config,
)
from regime_detection.cohort_routing import COHORTS, evaluate_cohort_routing
from regime_detection.engine import RegimeEngine
from regime_detection.models import AgentRouting, RegimeOutput  # noqa: F401
from regime_detection.network_fragility_rules import NetworkFragilityLabel  # noqa: F401
from regime_detection.trend_character import TrendCharacterLabel  # noqa: F401
from regime_detection.trend_direction import TrendDirectionLabel  # noqa: F401
from regime_detection.volatility_state import VolatilityLabel  # noqa: F401


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — real-config loader + a benign-baseline labelset builder so each
# test only varies the axis labels its rule cares about.
# ---------------------------------------------------------------------------


def _cohort_config() -> CohortRoutingConfig:
    """Load the cohort_routing block from the packaged default V2 yaml.

    Asserts the round-trip lands as a ``CohortRoutingConfig`` so any future
    yaml-shape regression surfaces at this single seam rather than 17 places.
    """
    cfg = load_default_regime_config().cohort_routing
    assert isinstance(cfg, CohortRoutingConfig), (
        "default V2 config must ship a cohort_routing block typed as CohortRoutingConfig"
    )
    return cfg


# Benign labelset: every axis at its lowest-risk real label. No specialist
# rule matches against this — used to isolate which axis a test varies.
_BENIGN: dict[str, str] = {
    "trend_direction_active": "sideways",          # TrendDirectionLabel (real)
    "trend_character_active": "trending",          # TrendCharacterLabel (real)
    "volatility_state_active": "normal_vol",       # VolatilityLabel (real)
    "breadth_state_active": "healthy_breadth",     # BreadthLabel (real)
    "network_fragility_active": "diversified_normal",  # NetworkFragilityLabel
    "monetary_pressure_active": None,              # §2A classifier not shipped
}


def _route(**overrides: object) -> AgentRouting:
    """Run ``evaluate_cohort_routing`` against the real default config with a
    benign baseline and the per-test overrides applied on top."""
    kwargs = {**_BENIGN, **overrides}
    return evaluate_cohort_routing(config=_cohort_config(), **kwargs)  # type: ignore[arg-type]


# ===========================================================================
# Group A — Precedence walker (one test per cohort)
# ===========================================================================


@pytest.mark.unit
def test_crisis_specialist_fires_on_correlation_to_one() -> None:
    """§5.1 lines 2517-2519: crisis fires when network_fragility is
    correlation_to_one even if every other axis is benign."""
    out = _route(network_fragility_active="correlation_to_one")
    assert out.active_cohort == "crisis_specialist"
    assert out.fallback_cohort == "default_neutral"
    assert out.blocked_strategy_modes == ["short_vol", "leveraged_long", "breakout"]


@pytest.mark.unit
def test_crisis_specialist_fires_on_systemic_stress() -> None:
    """§5.1 lines 2517-2519: systemic_stress also triggers crisis_specialist."""
    out = _route(network_fragility_active="systemic_stress")
    assert out.active_cohort == "crisis_specialist"
    assert out.blocked_strategy_modes == ["short_vol", "leveraged_long", "breakout"]


@pytest.mark.unit
def test_crisis_specialist_fires_on_unconfirmed_systemic_stress() -> None:
    """Unconfirmed systemic stress is crisis-equivalent for safety routing."""
    out = _route(network_fragility_active="systemic_stress_unconfirmed")
    assert out.active_cohort == "crisis_specialist"
    assert out.blocked_strategy_modes == ["short_vol", "leveraged_long", "breakout"]


@pytest.mark.unit
def test_crisis_specialist_fires_on_crisis_vol_alone() -> None:
    """§5.1 line 2519: ``OR volatility_state.active_label == 'crisis_vol'``
    — crisis_vol alone fires crisis even when network_fragility is benign."""
    out = _route(volatility_state_active="crisis_vol")
    assert out.active_cohort == "crisis_specialist"
    assert out.blocked_strategy_modes == ["short_vol", "leveraged_long", "breakout"]


@pytest.mark.unit
def test_crisis_outranks_bear_stress_when_both_match() -> None:
    """§5.1 precedence: crisis_specialist is listed before bear_stress, so
    a labelset matching both must route to crisis (defensive precedence)."""
    out = _route(
        trend_direction_active="bear",
        breadth_state_active="weak_breadth",
        volatility_state_active="crisis_vol",
    )
    assert out.active_cohort == "crisis_specialist"
    assert out.blocked_strategy_modes == ["short_vol", "leveraged_long", "breakout"]


@pytest.mark.unit
def test_bear_stress_specialist_fires_on_bear_and_weak_breadth() -> None:
    """§5.1 lines 2526-2528: bear_stress_specialist fires when
    trend=bear AND breadth in {weak_breadth, divergent_fragile,
    narrowing_breadth}. No crisis signals present → bear_stress wins."""
    out = _route(
        trend_direction_active="bear",
        breadth_state_active="weak_breadth",
    )
    assert out.active_cohort == "bear_stress_specialist"
    assert out.blocked_strategy_modes == ["short_vol", "breakout", "leveraged_long"]


@pytest.mark.unit
def test_bear_stress_does_not_fire_when_only_bear() -> None:
    """§5.1: bear alone with healthy_breadth does NOT match bear_stress
    (the rule is conjunctive). With no other specialist matching, the
    walker falls through to default_neutral."""
    out = _route(
        trend_direction_active="bear",
        breadth_state_active="healthy_breadth",
    )
    assert out.active_cohort == "default_neutral"
    assert out.blocked_strategy_modes == []


@pytest.mark.unit
def test_recovery_specialist_fires_on_trend_recovery() -> None:
    """§5.1 lines 2538-2539: recovery_specialist fires on
    trend_direction.active_label == 'recovery'."""
    out = _route(
        trend_direction_active="recovery",
        breadth_state_active="neutral_breadth",
        volatility_state_active="normal_vol",
    )
    assert out.active_cohort == "recovery_specialist"
    assert out.blocked_strategy_modes == ["short_vol"]


@pytest.mark.unit
def test_bull_low_vol_specialist_fires_on_bull_low_vol() -> None:
    """§5.1 lines 2548-2550: bull_low_vol fires on trend=bull AND
    vol in {low_vol, normal_vol}."""
    out = _route(
        trend_direction_active="bull",
        volatility_state_active="low_vol",
        breadth_state_active="healthy_breadth",
    )
    assert out.active_cohort == "bull_low_vol_specialist"
    assert out.blocked_strategy_modes == []


@pytest.mark.unit
def test_default_neutral_when_no_specialist_matches() -> None:
    """§5.1 line 2552: default_neutral matches when no specialist does.
    sideways + high_vol + healthy_breadth + diversified_normal matches
    none of the conjunctive specialist rules."""
    out = _route(
        trend_direction_active="sideways",
        trend_character_active="trending",
        volatility_state_active="high_vol",
        breadth_state_active="healthy_breadth",
        network_fragility_active="diversified_normal",
    )
    assert out.active_cohort == "default_neutral"
    assert out.blocked_strategy_modes == []


@pytest.mark.unit
def test_default_neutral_when_all_axes_unknown() -> None:
    """§5.1: silent specialists don't fire when their predicate labels are
    absent. All-unknown axes must route to default_neutral."""
    out = _route(
        trend_direction_active="unknown",
        trend_character_active="unknown",
        volatility_state_active="unknown",
        breadth_state_active="unknown",
        network_fragility_active="unknown",
        monetary_pressure_active=None,
    )
    assert out.active_cohort == "default_neutral"
    assert out.blocked_strategy_modes == []


# ===========================================================================
# Group B — Silent specialists (deferred-label predicates)
# ===========================================================================


@pytest.mark.unit
def test_euphoria_specialist_fires_when_trend_direction_is_euphoria() -> None:
    """§5.1 lines 2522-2523: euphoria_specialist's rule predicate is
    ``trend_direction.active_label == 'euphoria'``. The label is deferred
    until sentiment_score ships (Ambiguity Log #32), so in production this
    never fires — but the predicate is wired so the routing works the day
    sentiment lands. We pass the synthetic 'euphoria' string to verify."""
    out = _route(trend_direction_active="euphoria")
    assert out.active_cohort == "euphoria_specialist"
    assert out.blocked_strategy_modes == ["mean_reversion"]


@pytest.mark.unit
def test_tightening_specialist_silent_when_monetary_pressure_is_none() -> None:
    """§5.1 lines 2530-2531: tightening_specialist gates on monetary_pressure
    in {tightening_pressure, rate_shock}. With monetary_pressure_active=None
    (classifier not shipped per Ambiguity Log) the rule must NOT fire and
    the walker must fall through to default_neutral."""
    out = _route(monetary_pressure_active=None)
    assert out.active_cohort == "default_neutral"


@pytest.mark.unit
def test_tightening_specialist_fires_when_monetary_pressure_is_tightening_pressure() -> None:
    """§5.1 lines 2530-2531: when the future §2A classifier emits
    'tightening_pressure', the predicate must fire."""
    out = _route(monetary_pressure_active="tightening_pressure")
    assert out.active_cohort == "tightening_specialist"
    assert out.blocked_strategy_modes == []


# ===========================================================================
# Group C — Pydantic + config integration
# ===========================================================================


@pytest.mark.unit
def test_evaluate_cohort_routing_returns_agent_routing_with_fallback_default_neutral() -> None:
    """``fallback_cohort`` is always 'default_neutral' regardless of which
    specialist fires (spec §5.1: default_neutral is the universal fallback)."""
    for overrides in (
        {"network_fragility_active": "correlation_to_one"},      # crisis
        {"trend_direction_active": "bull", "volatility_state_active": "low_vol"},  # bull_low_vol
        {"trend_direction_active": "recovery"},                  # recovery
        {},                                                       # default_neutral
    ):
        out = _route(**overrides)  # type: ignore[arg-type]
        assert isinstance(out, AgentRouting)
        assert out.fallback_cohort == "default_neutral"
        assert out.active_cohort in COHORTS


@pytest.mark.unit
def test_evaluate_cohort_routing_uses_real_default_config() -> None:
    """End-to-end yaml round-trip: load_default_regime_config().cohort_routing
    must be a CohortRoutingConfig and must produce a valid AgentRouting."""
    cfg = load_default_regime_config().cohort_routing
    assert isinstance(cfg, CohortRoutingConfig)
    out = evaluate_cohort_routing(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="low_vol",
        breadth_state_active="healthy_breadth",
        network_fragility_active="diversified_normal",
        monetary_pressure_active=None,
        config=cfg,
    )
    assert out.active_cohort == "bull_low_vol_specialist"
    assert out.fallback_cohort == "default_neutral"
    assert out.blocked_strategy_modes == []


# ===========================================================================
# Group D — No legacy field aliases
# ===========================================================================


@pytest.mark.unit
def test_agent_routing_rejects_legacy_blocked_cohorts_field() -> None:
    """No backward-compatible alias: strategy-mode blocks are not cohorts."""
    with pytest.raises(ValidationError):
        AgentRouting.model_validate(
            {
                "active_cohort": "default_neutral",
                "fallback_cohort": "default_neutral",
                "blocked_cohorts": [],
            }
        )


@pytest.mark.unit
def test_cohort_routing_config_rejects_legacy_blocked_cohorts_field() -> None:
    """The config contract uses blocked_strategy_modes only."""
    cfg = _cohort_config()
    data = cfg.model_dump()
    data["blocked_cohorts"] = data.pop("blocked_strategy_modes")
    with pytest.raises(ValidationError):
        CohortRoutingConfig.model_validate(data)


# ===========================================================================
# Group E — Wire-in to RegimeOutput
# ===========================================================================


@pytest.mark.integration
def test_regime_output_emits_agent_routing_when_cohort_routing_configured(
    classified_golden_outputs: dict[date, RegimeOutput],
) -> None:
    """When the engine config carries a cohort_routing block (default V2),
    every classified output must populate ``agent_routing`` with one of the
    9 spec-pinned cohort names."""
    assert classified_golden_outputs, "golden outputs fixture must be non-empty"
    for as_of, out in classified_golden_outputs.items():
        assert out.agent_routing is not None, (
            f"agent_routing missing for {as_of}; default config carries cohort_routing"
        )
        assert out.agent_routing.active_cohort in COHORTS
        assert out.agent_routing.fallback_cohort == "default_neutral"


@pytest.mark.integration
def test_regime_output_omits_agent_routing_when_cohort_routing_absent_from_config(
    market_df_for_asof,
    golden_rows: list[dict[str, object]],
    synthetic_v2_kwargs_for_market_data,
) -> None:
    """V1 byte-identity preservation: a RegimeConfig with cohort_routing=None
    must produce ``RegimeOutput.agent_routing is None`` and the JSON dump
    must omit the field (exclude_none=True)."""
    engine = RegimeEngine()
    as_of = date.fromisoformat(str(golden_rows[0]["as_of_date"]))
    market_data = market_df_for_asof(as_of)
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    no_routing_config = kwargs["config"].model_copy(update={"cohort_routing": None})
    assert no_routing_config.cohort_routing is None
    kwargs["config"] = no_routing_config
    out = engine.classify(
        as_of_date=as_of,
        market_data=market_data,
        **kwargs,
    )
    assert out.agent_routing is None
    assert "agent_routing" not in out.model_dump()
