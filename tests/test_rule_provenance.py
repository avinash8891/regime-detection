from __future__ import annotations

from regime_detection.config import load_default_regime_config
from regime_detection.rule_provenance import (
    RULE_PROVENANCE,
    business_scalar_config_paths,
    provenance_by_key,
)


def test_rule_provenance_has_unique_keys_and_required_business_sections() -> None:
    by_key = provenance_by_key()

    assert len(by_key) == len(RULE_PROVENANCE)
    assert {
        "trend_direction.rules",
        "trend_character.rules",
        "volatility_state.rules",
        "breadth_state.rules",
        "network_fragility.rules",
        "volume_liquidity.rules",
        "monetary_pressure.rules",
        "inflation_growth.rules",
        "credit_funding.rules",
        "transition_score.weights",
        "event_calendar.precedence",
    }.issubset(by_key)


def test_rule_provenance_covers_every_business_scalar_config_path() -> None:
    cfg = load_default_regime_config()
    expected_paths = business_scalar_config_paths(cfg)
    provenance_paths = {entry.config_path for entry in RULE_PROVENANCE}

    assert expected_paths
    assert expected_paths <= provenance_paths


def test_rule_provenance_entries_are_mechanically_traceable() -> None:
    for entry in RULE_PROVENANCE:
        assert entry.config_path
        assert entry.spec_ref or entry.adr_refs
        assert entry.kind in {
            "threshold",
            "weight",
            "precedence",
            "hysteresis",
            "window",
            "staleness",
            "risk_rank",
            "input_contract",
            "model_parameter",
        }
