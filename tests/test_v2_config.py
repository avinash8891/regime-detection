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

from regime_detection import __version__
from regime_detection.config import (
    RegimeConfig,
    load_default_regime_config,
    load_regime_config,
)


# V2 spec §3.1 — canonical 22-asset network fragility universe (11 sector
# ETFs + SPY broad-market index + 10 cross-asset proxies). KRE belongs to
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
    "GLD",
    "HYG",
    "LQD",
    "USO",
    "UUP",
]

# V2 spec §3.7 — per-label asymmetric hysteresis deescalation days.
# `unknown: 5` added per Implementation Ambiguity Log entry #8: treat
# `unknown` as a high-risk hold so single-day quality flickers cannot
# fast-track de-escalation through the lower-risk band.
V2_NETWORK_FRAGILITY_DEESCALATION_DAYS = {
    "rising_fragility": 3,
    "correlation_concentration": 3,
    "correlation_to_one": 5,
    "systemic_stress": 5,
    "unknown": 5,
}

# V2 spec §3.5 — rule-engine thresholds (Slice 1.3). Each value cites a
# spec line verbatim. Used as a valid fixture for NetworkFragilityConfig
# construction in unit tests.
V2_NETWORK_FRAGILITY_RULES_KWARGS = dict(
    diversified_normal_percentile_lo=0.25,
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


def test_v2_default_config_has_22_etf_network_fragility_universe() -> None:
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    assert cfg.network_fragility.universe == V2_NETWORK_FRAGILITY_UNIVERSE
    assert len(cfg.network_fragility.universe) == 22


def test_v2_default_config_has_v2_section_3_7_deescalation_days() -> None:
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    assert (
        cfg.network_fragility.deescalation_days_by_label
        == V2_NETWORK_FRAGILITY_DEESCALATION_DAYS
    )


def test_v2_default_config_has_v2_section_3_2_lookback_windows() -> None:
    """v2 §3.2 — all six implementation lookback / completeness defaults
    must live in the config block so calibration (v2 §9.1) can tune them."""
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    nf = cfg.network_fragility
    assert nf.correlation_lookback_days == V2_NETWORK_FRAGILITY_CORRELATION_LOOKBACK_DAYS
    assert nf.percentile_lookback_days == V2_NETWORK_FRAGILITY_PERCENTILE_LOOKBACK_DAYS
    assert nf.realized_vol_lookback_days == V2_NETWORK_FRAGILITY_REALIZED_VOL_LOOKBACK_DAYS
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


def test_v2_transition_score_weights_with_hmm_sum_to_1_0() -> None:
    cfg = load_default_regime_config()
    assert cfg.transition_score is not None
    total = sum(cfg.transition_score.weights_with_hmm.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_v2_transition_score_weights_without_hmm_sum_to_1_0() -> None:
    cfg = load_default_regime_config()
    assert cfg.transition_score is not None
    total = sum(cfg.transition_score.weights_without_hmm.values())
    assert total == pytest.approx(1.0, abs=1e-9)


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


def test_v1_yaml_still_loads_with_v1_config_version() -> None:
    cfg = load_regime_config(_v1_yaml_path())
    assert cfg.config_version == "core3-v1.0.0"
    # All V2 sub-configs must remain None for the V1 yaml.
    assert cfg.network_fragility is None
    assert cfg.breadth_state_v2 is None
    assert cfg.transition_score is None
    assert cfg.monetary_pressure_v2 is None
    assert cfg.inflation_growth is None
    assert cfg.credit_funding is None
    assert cfg.event_calendar_v2 is None
    assert cfg.hmm is None
    assert cfg.vol_crush is None
    assert cfg.no_flip_flop is None
    assert cfg.cohort_routing is None
    assert cfg.strategy_family_constraints is None


def test_load_default_regime_config_dispatches_on_package_version() -> None:
    # Package is currently 2.x → default must be the V2 yaml.
    assert __version__.startswith("2.")
    cfg = load_default_regime_config()
    assert isinstance(cfg, RegimeConfig)
    assert cfg.config_version == "core3-v2.0.0"
