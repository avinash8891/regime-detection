from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

RuleProvenanceKind = Literal["threshold", "weight", "precedence", "hysteresis"]


class RuleProvenance(BaseModel):
    """Mechanical owner record for thresholds, weights, and rule precedence."""

    model_config = ConfigDict(extra="forbid")

    key: str
    owner: str
    kind: RuleProvenanceKind
    config_path: str
    spec_ref: str | None = None
    adr_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()


RULE_PROVENANCE: tuple[RuleProvenance, ...] = (
    RuleProvenance(
        key="trend_direction.rules",
        owner="trend_direction",
        kind="threshold",
        config_path="RegimeConfig.trend_direction",
        spec_ref="V1 §2.1 / V2 §1A",
        test_refs=("tests/test_schema_and_timeline.py",),
    ),
    RuleProvenance(
        key="trend_character.rules",
        owner="trend_character",
        kind="threshold",
        config_path="RegimeConfig.trend_character_v2",
        spec_ref="V1 §2.2 / V2 §1B",
        test_refs=("tests/test_trend_character.py",),
    ),
    RuleProvenance(
        key="volatility_state.rules",
        owner="volatility_state",
        kind="threshold",
        config_path="RegimeConfig.volatility_state_v2.rules",
        spec_ref="V1 §2.3 / V2 §1C",
        test_refs=("tests/test_volatility_state_v2_features.py",),
    ),
    RuleProvenance(
        key="breadth_state.rules",
        owner="breadth_state",
        kind="threshold",
        config_path="RegimeConfig.breadth_state_v2.rules",
        spec_ref="V1 §2.4 / V2 §1D",
        adr_refs=("ADR 0004",),
        test_refs=("tests/test_breadth_state_v2_labels.py",),
    ),
    RuleProvenance(
        key="network_fragility.rules",
        owner="network_fragility",
        kind="threshold",
        config_path="RegimeConfig.network_fragility.rules",
        spec_ref="V2 §3.5-§3.7",
        test_refs=("tests/test_network_fragility_classifier.py",),
    ),
    RuleProvenance(
        key="volume_liquidity.rules",
        owner="volume_liquidity_state",
        kind="threshold",
        config_path="RegimeConfig.volume_liquidity_v2.rules",
        spec_ref="V2 §1E",
        test_refs=("tests/test_volume_liquidity_classifier.py",),
    ),
    RuleProvenance(
        key="monetary_pressure.rules",
        owner="monetary_pressure_state",
        kind="threshold",
        config_path="RegimeConfig.monetary_pressure_v2.rules",
        spec_ref="V2 §2A",
        test_refs=("tests/test_monetary_pressure_classifier.py",),
    ),
    RuleProvenance(
        key="inflation_growth.rules",
        owner="inflation_growth_state",
        kind="threshold",
        config_path="RegimeConfig.inflation_growth.rules",
        spec_ref="V2 §2B",
        test_refs=("tests/test_inflation_growth_axis_engine.py",),
    ),
    RuleProvenance(
        key="credit_funding.rules",
        owner="credit_funding_state",
        kind="threshold",
        config_path="RegimeConfig.credit_funding.rules",
        spec_ref="V2 §2C",
        test_refs=("tests/test_credit_funding_axis_engine.py",),
    ),
    RuleProvenance(
        key="transition_score.weights",
        owner="transition_risk",
        kind="weight",
        config_path="RegimeConfig.transition_score",
        spec_ref="V2 §4.2-§4.3",
        test_refs=("tests/test_transition_risk.py",),
    ),
    RuleProvenance(
        key="event_calendar.precedence",
        owner="event_calendar",
        kind="precedence",
        config_path="RegimeConfig.event_calendar",
        spec_ref="V1 §2.6",
        test_refs=("tests/test_event_calendar.py",),
    ),
)


def provenance_by_key() -> dict[str, RuleProvenance]:
    by_key: dict[str, RuleProvenance] = {}
    for entry in RULE_PROVENANCE:
        if entry.key in by_key:
            raise RuntimeError(f"duplicate rule provenance key: {entry.key}")
        by_key[entry.key] = entry
    return by_key


def rule_provenance_payload() -> dict[str, dict[str, object]]:
    return {
        entry.key: entry.model_dump(mode="json", exclude={"key"}, exclude_none=True)
        for entry in RULE_PROVENANCE
    }
