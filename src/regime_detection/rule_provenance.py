from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from regime_detection.config import RegimeConfig, load_default_regime_config

RuleProvenanceKind = Literal[
    "threshold",
    "weight",
    "precedence",
    "hysteresis",
    "window",
    "staleness",
    "risk_rank",
    "input_contract",
    "model_parameter",
]


class RuleProvenance(BaseModel):
    """Mechanical owner record for business-logic scalar config and constants."""

    model_config = ConfigDict(extra="forbid")

    key: str
    owner: str
    kind: RuleProvenanceKind
    config_path: str
    spec_ref: str | None = None
    adr_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BusinessConfigRoot:
    path: str
    owner: str
    kind: RuleProvenanceKind
    spec_ref: str | None
    adr_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()


_BUSINESS_CONFIG_ROOTS: tuple[_BusinessConfigRoot, ...] = (
    _BusinessConfigRoot(
        "RegimeConfig.trend_direction",
        "trend_direction",
        "hysteresis",
        "V1 §2.1 / V2 §1A",
        test_refs=("tests/test_schema_and_timeline.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.trend_character",
        "trend_character",
        "hysteresis",
        "V1 §2.2 / V2 §1B",
        test_refs=("tests/test_trend_character.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.volatility_state",
        "volatility_state",
        "hysteresis",
        "V1 §2.3 / V2 §1C",
        test_refs=("tests/test_volatility_state_v2_features.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.breadth_state",
        "breadth_state",
        "hysteresis",
        "V1 §2.4 / V2 §1D",
        adr_refs=("ADR 0004",),
        test_refs=("tests/test_breadth_state_v2_labels.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.network_fragility",
        "network_fragility",
        "threshold",
        "V2 §3.1-§3.7",
        test_refs=("tests/test_network_fragility_classifier.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.trend_direction_v2",
        "trend_direction",
        "threshold",
        "V2 §1A",
        test_refs=("tests/test_trend_direction_v2_recovery_rule.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.volatility_state_v2",
        "volatility_state",
        "threshold",
        "V2 §1C",
        test_refs=("tests/test_volatility_state_v2_vol_crush.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.breadth_state_v2",
        "breadth_state",
        "threshold",
        "V2 §1D",
        adr_refs=("ADR 0004",),
        test_refs=("tests/test_breadth_state_v2_features.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.volume_liquidity_v2",
        "volume_liquidity",
        "threshold",
        "V2 §1E",
        test_refs=("tests/test_volume_liquidity_classifier.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.volume_liquidity_state",
        "volume_liquidity",
        "threshold",
        "V2 §1E",
        test_refs=("tests/test_volume_liquidity_classifier.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.transition_score",
        "transition_risk",
        "weight",
        "V2 §4.2-§4.4",
        test_refs=("tests/test_transition_risk.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.trend_character_v2",
        "trend_character",
        "threshold",
        "V2 §1B",
        test_refs=("tests/test_trend_character_v2_labels.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.monetary_pressure_v2",
        "monetary_pressure",
        "threshold",
        "V2 §2A",
        test_refs=("tests/test_monetary_pressure_classifier.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.monetary_pressure_state",
        "monetary_pressure",
        "threshold",
        "V2 §2A",
        test_refs=("tests/test_monetary_pressure_classifier.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.central_bank_text",
        "monetary_pressure",
        "input_contract",
        "V2 §2A",
        test_refs=("tests/test_central_bank_text.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.news_sentiment",
        "trend_direction",
        "input_contract",
        "V2 §1A",
        test_refs=("tests/test_trend_direction_v2_recovery_rule.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.inflation_growth",
        "inflation_growth",
        "threshold",
        "V2 §2B",
        adr_refs=("ADR 0011", "ADR 0012"),
        test_refs=("tests/test_inflation_growth_axis_engine.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.credit_funding",
        "credit_funding",
        "threshold",
        "V2 §2C",
        test_refs=("tests/test_credit_funding_axis_engine.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.hmm",
        "hmm",
        "model_parameter",
        "V2 §6.1",
        adr_refs=("ADR 0013",),
        test_refs=("tests/test_hmm_state.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.clustering",
        "clustering",
        "model_parameter",
        "V2 §6.2",
        adr_refs=("ADR 0013",),
        test_refs=("tests/test_clustering.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.change_point",
        "change_point",
        "model_parameter",
        "V2 §6.3",
        adr_refs=("ADR 0013",),
        test_refs=("tests/test_change_point.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.no_flip_flop",
        "no_flip_flop",
        "window",
        "V2 §5.4",
        test_refs=("tests/test_no_flip_flop.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.cohort_routing",
        "cohort_routing",
        "precedence",
        "V2 §5.1",
        test_refs=("tests/test_cohort_routing.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.strategy_family_constraints",
        "strategy_family_constraints",
        "input_contract",
        "V2 §5.2",
        test_refs=("tests/test_strategy_family_constraints.py",),
    ),
    _BusinessConfigRoot(
        "RegimeConfig.strategy_event_modifiers",
        "strategy_event_modifiers",
        "input_contract",
        "V2 §5.3",
        test_refs=("tests/test_strategy_response.py",),
    ),
)


_SECTION_PROVENANCE: tuple[RuleProvenance, ...] = (
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
        config_path="RegimeConfig.breadth_state_v2",
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
        owner="volume_liquidity",
        kind="threshold",
        config_path="RegimeConfig.volume_liquidity_state.rules",
        spec_ref="V2 §1E",
        test_refs=("tests/test_volume_liquidity_classifier.py",),
    ),
    RuleProvenance(
        key="monetary_pressure.rules",
        owner="monetary_pressure",
        kind="threshold",
        config_path="RegimeConfig.monetary_pressure_state.rules",
        spec_ref="V2 §2A",
        test_refs=("tests/test_monetary_pressure_classifier.py",),
    ),
    RuleProvenance(
        key="inflation_growth.rules",
        owner="inflation_growth",
        kind="threshold",
        config_path="RegimeConfig.inflation_growth.rules",
        spec_ref="V2 §2B",
        test_refs=("tests/test_inflation_growth_axis_engine.py",),
    ),
    RuleProvenance(
        key="credit_funding.rules",
        owner="credit_funding",
        kind="threshold",
        config_path="RegimeConfig.credit_funding.rules",
        spec_ref="V2 §2C",
        test_refs=("tests/test_credit_funding_axis_engine.py",),
    ),
    RuleProvenance(
        key="transition_score.weights",
        owner="transition_risk",
        kind="weight",
        config_path="RegimeConfig.transition_score.weights",
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


_STATIC_PROVENANCE: tuple[RuleProvenance, ...] = (
    RuleProvenance(
        key="trend_direction.precedence",
        owner="trend_direction",
        kind="precedence",
        config_path="trend_direction_v2.TREND_DIRECTION_V2_PRECEDENCE",
        spec_ref="V2 §1A line 239",
        test_refs=("tests/test_trend_direction_v2_recovery_rule.py",),
    ),
    RuleProvenance(
        key="network_fragility.precedence",
        owner="network_fragility",
        kind="precedence",
        config_path="network_fragility_rules.evaluate_rules",
        spec_ref="V2 §3.4-§3.5",
        test_refs=("tests/test_network_fragility_rules.py",),
    ),
    RuleProvenance(
        key="inflation_growth.precedence",
        owner="inflation_growth",
        kind="precedence",
        config_path="inflation_growth.evaluate_rules",
        spec_ref="V2 §2B line 2980",
        adr_refs=("ADR 0011", "ADR 0012"),
        test_refs=("tests/test_inflation_growth_axis_engine.py",),
    ),
    RuleProvenance(
        key="credit_funding.precedence",
        owner="credit_funding",
        kind="precedence",
        config_path="credit_funding.evaluate_rules",
        spec_ref="V2 §2C line 3183",
        test_refs=("tests/test_credit_funding_axis_engine.py",),
    ),
    RuleProvenance(
        key="credit_funding.risk_rank",
        owner="credit_funding",
        kind="risk_rank",
        config_path="credit_funding.CREDIT_FUNDING_RISK_RANK",
        spec_ref="V2 §2C lines 3277-3283",
        test_refs=("tests/test_credit_funding_axis_engine.py",),
    ),
    RuleProvenance(
        key="inflation_growth.risk_rank",
        owner="inflation_growth",
        kind="risk_rank",
        config_path="inflation_growth.INFLATION_GROWTH_RISK_RANK",
        spec_ref="V2 §2B lines 3109-3124",
        adr_refs=("ADR 0011", "ADR 0012"),
        test_refs=("tests/test_inflation_growth_axis_engine.py",),
    ),
)


def business_scalar_config_paths(config: RegimeConfig) -> set[str]:
    """Return every scalar business-logic config path requiring provenance."""

    paths: set[str] = set()
    for root in _BUSINESS_CONFIG_ROOTS:
        value = _value_at_config_path(config, root.path)
        if value is None:
            continue
        paths.update(_iter_scalar_paths(value, root.path))
    return paths


def _value_at_config_path(config: RegimeConfig, path: str) -> object:
    value: object = config
    for part in path.removeprefix("RegimeConfig.").split("."):
        value = getattr(value, part)
    return value


def _iter_scalar_paths(value: object, prefix: str) -> set[str]:
    if _is_scalar(value):
        return {prefix}
    if isinstance(value, BaseModel):
        paths: set[str] = set()
        for field_name in value.__class__.model_fields:
            child = getattr(value, field_name)
            if child is None:
                continue
            paths.update(_iter_scalar_paths(child, f"{prefix}.{field_name}"))
        return paths
    if isinstance(value, dict):
        paths = set()
        for key, child in cast(dict[object, object], value).items():
            if child is None:
                continue
            paths.update(_iter_scalar_paths(child, f"{prefix}.{key}"))
        return paths
    if isinstance(value, tuple):
        paths = set()
        for idx, child in enumerate(cast(tuple[object, ...], value)):
            if child is None:
                continue
            paths.update(_iter_scalar_paths(child, f"{prefix}.{idx}"))
        return paths
    return set()


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def _root_for_path(path: str) -> _BusinessConfigRoot:
    matches = [root for root in _BUSINESS_CONFIG_ROOTS if path.startswith(root.path)]
    if not matches:
        raise RuntimeError(f"no business provenance root for scalar path: {path}")
    return max(matches, key=lambda root: len(root.path))


def _kind_for_scalar_path(path: str, default: RuleProvenanceKind) -> RuleProvenanceKind:
    leaf = path.rsplit(".", 1)[-1]
    lowered = path.lower()
    if ".weights." in lowered or leaf == "weights":
        return "weight"
    if "deescalation" in lowered or "hysteresis" in lowered:
        return "hysteresis"
    if "stale" in lowered or "freshness" in lowered or "completeness" in lowered:
        return "staleness"
    if any(
        token in lowered
        for token in (
            "lookback",
            "window",
            "period",
            "days",
            "sessions",
            "cadence",
            "confirmation",
        )
    ):
        return "window"
    if any(
        token in lowered
        for token in (
            "threshold",
            "percentile",
            "ratio",
            "min",
            "max",
            "floor",
            "range",
            "ceiling",
            "band",
            "scale",
            "zscore",
            "score",
            "drop",
            "drawdown",
        )
    ):
        return "threshold"
    return default


def _scalar_key(path: str) -> str:
    return "config." + path.removeprefix("RegimeConfig.")


def _scalar_provenance(config: RegimeConfig) -> tuple[RuleProvenance, ...]:
    rows: list[RuleProvenance] = []
    for path in sorted(business_scalar_config_paths(config)):
        root = _root_for_path(path)
        rows.append(
            RuleProvenance(
                key=_scalar_key(path),
                owner=root.owner,
                kind=_kind_for_scalar_path(path, root.kind),
                config_path=path,
                spec_ref=root.spec_ref,
                adr_refs=root.adr_refs,
                test_refs=root.test_refs,
            )
        )
    return tuple(rows)


def _dedupe(entries: tuple[RuleProvenance, ...]) -> tuple[RuleProvenance, ...]:
    by_key: dict[str, RuleProvenance] = {}
    for entry in entries:
        if entry.key in by_key:
            raise RuntimeError(f"duplicate rule provenance key: {entry.key}")
        by_key[entry.key] = entry
    return tuple(by_key.values())


def build_rule_provenance(
    config: RegimeConfig | None = None,
) -> tuple[RuleProvenance, ...]:
    cfg = config if config is not None else load_default_regime_config()
    return _dedupe(_SECTION_PROVENANCE + _STATIC_PROVENANCE + _scalar_provenance(cfg))


RULE_PROVENANCE: tuple[RuleProvenance, ...] = build_rule_provenance()


def provenance_by_key() -> dict[str, RuleProvenance]:
    by_key: dict[str, RuleProvenance] = {}
    for entry in RULE_PROVENANCE:
        if entry.key in by_key:
            raise RuntimeError(f"duplicate rule provenance key: {entry.key}")
        by_key[entry.key] = entry
    return by_key


def rule_provenance_payload() -> dict[str, dict[str, Any]]:
    return {
        entry.key: entry.model_dump(mode="json", exclude={"key"}, exclude_none=True)
        for entry in RULE_PROVENANCE
    }
