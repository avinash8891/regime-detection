"""Phase B — V2 config schema + yaml tests.

References:
    docs/regime_engine_v2_spec.md §3 (network fragility), §3.7 (per-label
    deescalation days), §4.3 (transition score weights), §4.4 (bands).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typing import get_args

from regime_detection import __version__
import regime_detection.config as config_module
from regime_detection.config import (
    RegimeConfig,
    _default_config_resource_name_for_version,
    load_default_regime_config,
    load_regime_config,
)
from regime_detection.model_status import ClassificationStatus

# V2 spec §3.1 — canonical 24-asset network fragility universe (11 sector
# ETFs + SPY broad-market index + 12 cross-asset proxies). KRE belongs to
# v2 §2C credit/funding, not §3.1.
V2_NETWORK_FRAGILITY_UNIVERSE = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "GLD",
    "HYG",
    "LQD",
    "USO",
    "DBC",
    "UUP",
]

# V2 spec §3.7 — per-label asymmetric hysteresis deescalation days.
# `unknown: 0` keeps absence-of-signal from delaying recovery into a valid
# classified label. Flickers from high-risk labels are still held by the
# high-risk label's own deescalation threshold.
V2_NETWORK_FRAGILITY_DEESCALATION_DAYS = {
    "rising_fragility": 3,
    "correlation_concentration": 3,
    "correlation_to_one": 5,
    "systemic_stress_unconfirmed": 5,
    "systemic_stress": 5,
    "idiosyncratic_crisis": 3,
    "rotation_watch": 0,
    "decorrelated_calm": 0,
    "unknown": 0,
}

# V2 spec §3.5 — rule-engine thresholds (Slice 1.3). Each value cites a
# spec line verbatim. Used as a valid fixture for NetworkFragilityConfig
# construction in unit tests.
V2_NETWORK_FRAGILITY_RULES_KWARGS = dict(
    diversified_normal_percentile_lo=0.0,
    diversified_normal_percentile_hi=0.75,
    effective_rank_stability_threshold=0.05,
    stock_picker_percentile_max=0.30,
    stock_picker_dispersion_percentile_min=0.70,
    concentration_corr_percentile_min=0.75,
    concentration_largest_eig_percentile_min=0.75,
    concentration_effective_rank_percentile_max=0.25,
    corr_to_one_corr_percentile_min=0.90,
    corr_to_one_realized_vol_percentile_min=0.80,
    corr_to_one_drawdown_max=0.0,
    systemic_stress_vix_percentile_min=0.80,
)

# v2 spec §3.2 — implementation lookback windows hoisted to config so they
# remain calibration-tunable per slice-gate checklist §2.
V2_NETWORK_FRAGILITY_CORRELATION_LOOKBACK_DAYS = 63
V2_NETWORK_FRAGILITY_PERCENTILE_LOOKBACK_DAYS = 504
V2_NETWORK_FRAGILITY_REALIZED_VOL_LOOKBACK_DAYS = 21
V2_NETWORK_FRAGILITY_DISPERSION_PERCENTILE_LOOKBACK_DAYS = 252
V2_NETWORK_FRAGILITY_MIN_UNIVERSE_SIZE = 20
V2_NETWORK_FRAGILITY_MIN_WINDOW_COMPLETENESS = 0.90


def _v1_yaml_path() -> Path:
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v1.0.0.yaml"
    )
    # importlib.resources.files returns a Traversable; cast through str for load_regime_config.
    return Path(str(pkg_file))


def test_v2_default_config_loads_and_has_correct_version() -> None:
    cfg = load_default_regime_config()
    assert cfg.config_version == "core3-v2.0.0"


def test_v2_default_config_has_24_etf_network_fragility_universe() -> None:
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    assert cfg.network_fragility.universe == V2_NETWORK_FRAGILITY_UNIVERSE
    assert len(cfg.network_fragility.universe) == 24


def test_v2_default_config_has_v2_section_3_7_deescalation_days() -> None:
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    assert (
        cfg.network_fragility.deescalation_days_by_label
        == V2_NETWORK_FRAGILITY_DEESCALATION_DAYS
    )


def test_v2_spec_classification_status_table_matches_shipped_statuses() -> None:
    spec = Path("docs/regime_engine_v2_spec.md").read_text()

    for status in get_args(ClassificationStatus):
        assert f"`{status}`" in spec


def test_v2_spec_formalizes_systemic_stress_unconfirmed_label() -> None:
    spec = Path("docs/regime_engine_v2_spec.md").read_text()

    assert "`systemic_stress_unconfirmed`:" in spec
    assert "systemic_stress_unconfirmed: 3" in spec
    assert "systemic_stress_unconfirmed: 5" in spec
    assert "Resolved by Slice 2.8c status update" in spec


def test_layer1_axis_hysteresis_lives_on_axis_sections_not_v2_feature_configs() -> None:
    cfg = load_default_regime_config()

    assert cfg.trend_direction.default_escalation_days == 1
    assert cfg.trend_character.default_escalation_days == 1
    assert cfg.volatility_state.default_escalation_days == 1
    assert cfg.breadth_state.default_escalation_days == 1
    assert cfg.trend_direction.escalation_days_by_label == {}
    assert cfg.trend_direction.deescalation_days_by_label["bear"] == 5
    assert cfg.trend_direction.deescalation_days_by_label["unknown"] == 0
    assert cfg.trend_character.deescalation_days_by_label["range_bound"] == 3
    assert cfg.trend_character.deescalation_days_by_label["unknown"] == 0
    assert cfg.volatility_state.deescalation_days_by_label["crisis_vol"] == 5
    assert cfg.volatility_state.deescalation_days_by_label["unknown"] == 0
    assert cfg.breadth_state.deescalation_days_by_label["divergent_fragile"] == 5
    assert cfg.breadth_state.deescalation_days_by_label["unknown"] == 0


def test_v2_default_config_declares_unknown_freeze_windows() -> None:
    cfg = load_default_regime_config()

    assert cfg.trend_direction.max_unknown_freeze_days == 2
    assert cfg.trend_character.max_unknown_freeze_days == 2
    assert cfg.volatility_state.max_unknown_freeze_days == 2
    assert cfg.breadth_state.max_unknown_freeze_days == 2
    assert cfg.network_fragility is not None
    assert cfg.network_fragility.max_unknown_freeze_days == 2
    assert cfg.volume_liquidity_state is not None
    assert cfg.volume_liquidity_state.max_unknown_freeze_days == 2
    assert cfg.monetary_pressure_state is not None
    assert cfg.monetary_pressure_state.max_unknown_freeze_days == 2
    assert cfg.credit_funding is not None
    assert cfg.credit_funding.max_unknown_freeze_days == 2
    assert cfg.inflation_growth is not None
    assert cfg.inflation_growth.max_unknown_freeze_days == 2


def test_disinflation_yield_independent_policy_is_explicit_in_config_and_docs() -> None:
    """ADR 0011 is the authority for yield-independent disinflation."""
    cfg = load_default_regime_config()

    assert cfg.inflation_growth is not None
    assert cfg.inflation_growth.rules.disinflation_yield_independent is True

    spec = Path("docs/regime_engine_v2_spec.md").read_text()
    adr = Path("docs/decisions/0011-inflation-growth-rule-coverage-fix.md").read_text()
    authority = "Yield-independent disinflation is the default production policy"
    assert authority in spec
    assert authority in adr


@pytest.mark.parametrize(
    "section",
    [
        "trend_direction_v2",
        "trend_character_v2",
        "volatility_state_v2",
        "breadth_state_v2",
    ],
)
def test_layer1_v2_configs_reject_dead_hysteresis_knobs(
    tmp_path: Path, section: str
) -> None:
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v2.0.0.yaml"
    )
    data = yaml.safe_load(pkg_file.read_text(encoding="utf-8"))
    data[section]["deescalation_days_by_label"] = {"unknown": 9}
    data[section]["default_deescalation_days"] = 9

    bad_yaml = tmp_path / f"core3-v2.0.0-dead-{section}-hysteresis.yaml"
    bad_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_regime_config(bad_yaml)


def test_v2_default_config_has_v2_section_3_2_lookback_windows() -> None:
    """v2 §3.2 — all six implementation lookback / completeness defaults
    must live in the config block so calibration (v2 §9.1) can tune them."""
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    nf = cfg.network_fragility
    assert (
        nf.correlation_lookback_days == V2_NETWORK_FRAGILITY_CORRELATION_LOOKBACK_DAYS
    )
    assert nf.percentile_lookback_days == V2_NETWORK_FRAGILITY_PERCENTILE_LOOKBACK_DAYS
    assert (
        nf.realized_vol_lookback_days == V2_NETWORK_FRAGILITY_REALIZED_VOL_LOOKBACK_DAYS
    )
    assert (
        nf.dispersion_percentile_lookback_days
        == V2_NETWORK_FRAGILITY_DISPERSION_PERCENTILE_LOOKBACK_DAYS
    )
    assert nf.min_universe_size == V2_NETWORK_FRAGILITY_MIN_UNIVERSE_SIZE
    assert nf.min_window_completeness == V2_NETWORK_FRAGILITY_MIN_WINDOW_COMPLETENESS


def test_network_fragility_config_forbids_extra_fields() -> None:
    """Slice-gate checklist §2 — NetworkFragilityConfig must enforce
    extra='forbid' so unknown keys raise on load."""
    from regime_detection.config import NetworkFragilityConfig

    valid_kwargs = dict(
        universe=V2_NETWORK_FRAGILITY_UNIVERSE,
        correlation_lookback_days=V2_NETWORK_FRAGILITY_CORRELATION_LOOKBACK_DAYS,
        percentile_lookback_days=V2_NETWORK_FRAGILITY_PERCENTILE_LOOKBACK_DAYS,
        realized_vol_lookback_days=V2_NETWORK_FRAGILITY_REALIZED_VOL_LOOKBACK_DAYS,
        dispersion_percentile_lookback_days=V2_NETWORK_FRAGILITY_DISPERSION_PERCENTILE_LOOKBACK_DAYS,
        min_universe_size=V2_NETWORK_FRAGILITY_MIN_UNIVERSE_SIZE,
        min_window_completeness=V2_NETWORK_FRAGILITY_MIN_WINDOW_COMPLETENESS,
        deescalation_days_by_label=V2_NETWORK_FRAGILITY_DEESCALATION_DAYS,
        rules=V2_NETWORK_FRAGILITY_RULES_KWARGS,
    )
    # Sanity: valid kwargs construct cleanly.
    NetworkFragilityConfig(**valid_kwargs)

    # Extra field must raise.
    with pytest.raises(ValidationError):
        NetworkFragilityConfig(**valid_kwargs, unknown_calibration_knob=42)


def test_network_fragility_config_rejects_invalid_lookback_bounds() -> None:
    """v2 §3.2 lookback windows must be positive ints; completeness in [0,1]."""
    from regime_detection.config import NetworkFragilityConfig

    base = dict(
        universe=V2_NETWORK_FRAGILITY_UNIVERSE,
        correlation_lookback_days=V2_NETWORK_FRAGILITY_CORRELATION_LOOKBACK_DAYS,
        percentile_lookback_days=V2_NETWORK_FRAGILITY_PERCENTILE_LOOKBACK_DAYS,
        realized_vol_lookback_days=V2_NETWORK_FRAGILITY_REALIZED_VOL_LOOKBACK_DAYS,
        dispersion_percentile_lookback_days=V2_NETWORK_FRAGILITY_DISPERSION_PERCENTILE_LOOKBACK_DAYS,
        min_universe_size=V2_NETWORK_FRAGILITY_MIN_UNIVERSE_SIZE,
        min_window_completeness=V2_NETWORK_FRAGILITY_MIN_WINDOW_COMPLETENESS,
        deescalation_days_by_label=V2_NETWORK_FRAGILITY_DEESCALATION_DAYS,
        rules=V2_NETWORK_FRAGILITY_RULES_KWARGS,
    )

    # realized_vol_lookback_days must be > 0
    bad = {**base, "realized_vol_lookback_days": 0}
    with pytest.raises(ValidationError):
        NetworkFragilityConfig(**bad)

    # dispersion_percentile_lookback_days must be > 0
    bad = {**base, "dispersion_percentile_lookback_days": -1}
    with pytest.raises(ValidationError):
        NetworkFragilityConfig(**bad)

    # min_window_completeness must be in [0,1]
    bad = {**base, "min_window_completeness": 1.5}
    with pytest.raises(ValidationError):
        NetworkFragilityConfig(**bad)


def test_v2_transition_score_weights_sum_to_1_0() -> None:
    cfg = load_default_regime_config()
    assert cfg.transition_score is not None
    total = sum(cfg.transition_score.weights.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_v2_transition_score_config_uses_score_first_component_contract() -> None:
    cfg = load_default_regime_config()
    assert cfg.transition_score is not None
    assert set(cfg.transition_score.weights) == {
        "trend_break",
        "volatility_acceleration",
        "breadth_deterioration",
        "correlation_fragility",
        "credit_stress",
        "liquidity_stress",
        "macro_event",
        "model_instability",
    }
    assert cfg.transition_score.minimum_component_weight_coverage == pytest.approx(0.75)
    assert cfg.transition_score.bands == {
        "stable": (0.0, 0.35),
        "weakening": (0.35, 0.55),
        "transition_warning": (0.55, 0.75),
        "high": (0.75, 1.0),
    }
    assert cfg.transition_score.state_confirmation_days == {
        "stable": 1,
        "watch": 1,
        "weakening": 2,
        "transition_warning": 2,
        "high_transition_risk": 2,
        "fragile_bull": 2,
        "recovery_attempt": 2,
        "bear_stress": 1,
        "crisis": 1,
        "insufficient_data": 1,
    }


def test_v2_default_config_loads_strategy_event_modifiers() -> None:
    cfg = load_default_regime_config()

    assert cfg.strategy_event_modifiers is not None
    rules = cfg.strategy_event_modifiers.rules
    assert set(rules) == {"macro_event_window", "policy_or_event_risk_window"}

    macro = rules["macro_event_window"]
    assert macro.labels == (
        "fed_week",
        "cpi_week",
        "nfp_week",
        "global_rate_decision",
    )
    assert macro.position_size_cap == pytest.approx(0.75)
    assert macro.allow_leverage_expansion is False
    assert macro.require_confirmation_for_new_longs is True
    assert macro.leverage_allowed is None
    assert macro.prefer_cash_or_hedges is None

    policy = rules["policy_or_event_risk_window"]
    assert policy.labels == ("budget_week", "election_window", "geopolitical_event")
    assert policy.position_size_cap == pytest.approx(0.50)
    assert policy.leverage_allowed is False
    assert policy.prefer_cash_or_hedges is True
    assert policy.require_confirmation_for_new_longs is True
    assert policy.allow_leverage_expansion is None


def test_v2_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    # Start from the packaged v2 yaml and inject an unknown top-level field.
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v2.0.0.yaml"
    )
    data = yaml.safe_load(pkg_file.read_text(encoding="utf-8"))
    data["unknown_v2_axis"] = {"foo": "bar"}

    bad_yaml = tmp_path / "core3-v2.0.0-with-unknown.yaml"
    bad_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_regime_config(bad_yaml)


def test_v2_config_rejects_volume_liquidity_without_volatility_v2(
    tmp_path: Path,
) -> None:
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v2.0.0.yaml"
    )
    data = yaml.safe_load(pkg_file.read_text(encoding="utf-8"))
    data["volatility_state_v2"] = None

    bad_yaml = tmp_path / "core3-v2.0.0-without-volatility-v2.yaml"
    bad_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(
        ValidationError,
        match="missing required V2 sections: volatility_state_v2",
    ):
        load_regime_config(bad_yaml)


def test_v2_config_rejects_volume_liquidity_state_without_volume_liquidity_v2(
    tmp_path: Path,
) -> None:
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v2.0.0.yaml"
    )
    data = yaml.safe_load(pkg_file.read_text(encoding="utf-8"))
    data["volume_liquidity_v2"] = None

    bad_yaml = tmp_path / "core3-v2.0.0-without-volume-liquidity-v2.yaml"
    bad_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(
        ValidationError,
        match="volume_liquidity_state requires volume_liquidity_v2",
    ):
        load_regime_config(bad_yaml)


def test_v1_yaml_still_loads_with_v1_config_version() -> None:
    cfg = load_regime_config(_v1_yaml_path())
    assert cfg.config_version == "core3-v1.0.0"
    # L1 axes require per-label hysteresis in both V1 and V2.
    assert cfg.trend_direction.deescalation_days_by_label["bear"] == 3
    assert cfg.volatility_state.deescalation_days_by_label["crisis_vol"] == 2
    assert cfg.breadth_state.deescalation_days_by_label["weak_breadth"] == 2
    assert cfg.trend_character.deescalation_days_by_label["trending"] == 3
    assert not (
        set(cfg.trend_character.deescalation_days_by_label)
        & {"breakout_expansion", "mild_trend", "range_bound", "volatile_chop"}
    )
    # V2-only sub-configs must remain None for the V1 yaml.
    assert cfg.network_fragility is None
    assert cfg.transition_score is None
    assert cfg.monetary_pressure_v2 is None
    assert cfg.inflation_growth is None
    assert cfg.credit_funding is None
    assert cfg.hmm is None
    assert cfg.no_flip_flop is None
    assert cfg.cohort_routing is None
    assert cfg.strategy_family_constraints is None
    assert cfg.strategy_event_modifiers is None


def test_v1_yaml_does_not_enable_v2_feature_blocks() -> None:
    cfg = load_regime_config(_v1_yaml_path())

    assert cfg.trend_direction_v2 is None
    assert cfg.volatility_state_v2 is None
    assert cfg.breadth_state_v2 is None
    assert cfg.trend_character_v2 is None


def test_strategy_event_modifier_config_rejects_unknown_label() -> None:
    from regime_detection.config import StrategyEventModifierRule

    with pytest.raises(ValidationError, match="labels"):
        StrategyEventModifierRule(
            labels=("fomc_surprise",),
            position_size_cap=0.75,
        )


def test_strategy_event_modifier_config_uses_runtime_event_label_source() -> None:
    from regime_detection.config import StrategyEventModifierRule
    from regime_detection.event_calendar_labels import EVENT_CALENDAR_LABELS

    rule = StrategyEventModifierRule(
        labels=EVENT_CALENDAR_LABELS,
        position_size_cap=0.75,
    )

    assert rule.labels == EVENT_CALENDAR_LABELS


def test_strategy_event_modifier_config_rejects_rule_with_no_action_fields() -> None:
    from regime_detection.config import StrategyEventModifierRule

    with pytest.raises(ValidationError, match="at least one action"):
        StrategyEventModifierRule(labels=("fed_week",))


@pytest.mark.parametrize(
    "weakening_action",
    [
        {"leverage_allowed": True},
        {"allow_leverage_expansion": True},
        {"require_confirmation_for_new_longs": False},
        {"prefer_cash_or_hedges": False},
    ],
)
def test_strategy_event_modifier_config_rejects_risk_loosening_actions(
    weakening_action: dict[str, bool],
) -> None:
    from regime_detection.config import StrategyEventModifierRule

    with pytest.raises(ValidationError, match="cannot loosen"):
        StrategyEventModifierRule(labels=("fed_week",), **weakening_action)


def test_load_default_regime_config_dispatches_on_package_version() -> None:
    # Package is currently 2.x → default must be the V2 yaml.
    assert __version__.startswith("2.")
    cfg = load_default_regime_config()
    assert isinstance(cfg, RegimeConfig)
    assert cfg.config_version == "core3-v2.0.0"
    assert cfg.no_flip_flop is not None
    assert cfg.no_flip_flop.window_trading_days == 15


def test_default_config_resource_dispatch_parses_version_major() -> None:
    assert (
        _default_config_resource_name_for_version("1.9.9")
        == "configs/core3-v1.0.0.yaml"
    )
    assert (
        _default_config_resource_name_for_version("2.0.0rc1")
        == "configs/core3-v2.0.0.yaml"
    )


def test_default_config_resource_dispatch_rejects_unsupported_major() -> None:
    with pytest.raises(ValueError, match="Unsupported package __version__"):
        _default_config_resource_name_for_version("20.0.0")


def test_load_default_regime_config_uses_parsed_version_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "__version__", "2.0.0rc1")

    cfg = load_default_regime_config()

    assert cfg.config_version == "core3-v2.0.0"
