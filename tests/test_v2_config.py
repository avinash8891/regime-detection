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


# V2 spec §3.1 — canonical 22-ETF network fragility universe.
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
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
    "UUP",
    "KRE",
]

# V2 spec §3.7 — per-label asymmetric hysteresis deescalation days.
V2_NETWORK_FRAGILITY_DEESCALATION_DAYS = {
    "rising_fragility": 3,
    "correlation_concentration": 3,
    "correlation_to_one": 5,
    "systemic_stress": 5,
}


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
    assert cfg.transition_score is None
    assert cfg.monetary_pressure_v2 is None
    assert cfg.inflation_growth is None
    assert cfg.credit_funding is None
    assert cfg.event_calendar_v2 is None
    assert cfg.hmm is None
    assert cfg.vol_crush is None
    assert cfg.no_flip_flop is None
    assert cfg.strategy_cohort is None
    assert cfg.strategy_family_constraints is None


def test_load_default_regime_config_dispatches_on_package_version() -> None:
    # Package is currently 2.x → default must be the V2 yaml.
    assert __version__.startswith("2.")
    cfg = load_default_regime_config()
    assert isinstance(cfg, RegimeConfig)
    assert cfg.config_version == "core3-v2.0.0"
