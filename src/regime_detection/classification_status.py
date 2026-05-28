from __future__ import annotations

import math
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from regime_detection.evidence_payloads import EvidencePayload
from regime_detection.model_status import ClassificationStatus, DataQualityStatus

_NON_BINDING_MISSING_RULE_FEATURES = {
    "broad_usd_index_zscore_21d",
    "inflation_surprise_zscore",
    "aggregate_forward_eps_revision_direction_4w",
}


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatus
    freshness_days: int | None = Field(default=None, ge=0)
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


def derive_classification_status(
    *,
    active_label: str,
    data_quality: DataQuality,
    evidence: Any | None = None,
    raw_label: str | None = None,
    stable_label: str | None = None,
) -> tuple[ClassificationStatus, str | None]:
    """Disambiguate legacy ``unknown`` labels from data-quality failures.

    ``active_label`` remains the backward-compatible regime label. This helper
    adds the semantic reason a label was emitted, so reports can distinguish
    "data was unavailable" from "data was usable but no rule matched".
    """
    evidence_reason = None
    if evidence is not None:
        raw_reason = evidence.get("reason")
        if isinstance(raw_reason, str) and raw_reason:
            evidence_reason = raw_reason

    reason = data_quality.reason or evidence_reason
    if data_quality.status == "stale_data":
        return "stale_data", reason or "stale_data"
    if data_quality.status == "insufficient_history":
        return "insufficient_history", reason or "insufficient_history"
    if data_quality.status == "insufficient_data":
        return "data_unavailable", reason or "insufficient_data"
    if active_label != "unknown":
        return "classified", None
    if raw_label not in {None, "unknown"} or stable_label not in {None, "unknown"}:
        return "no_rule_fired_hysteresis", "hysteresis_held_unknown"
    missing_rule_features = _missing_rule_features(evidence)
    if missing_rule_features:
        return "no_rule_fired_missing_feature", _missing_rule_feature_reason(
            missing_rule_features
        )
    return "no_rule_fired", reason or "no_rule_fired"


def _missing_rule_features(evidence: Any | None) -> list[str]:
    if evidence is None:
        return []
    features: set[str] = set()
    _collect_missing_rule_features(evidence, features)
    return sorted(features)


def _missing_rule_feature_reason(features: list[str]) -> str:
    prefix = "missing_rule_feature" if len(features) == 1 else "missing_rule_features"
    return f"{prefix}:{','.join(features)}"


def _collect_missing_rule_features(value: Any, features: set[str]) -> None:
    if isinstance(value, EvidencePayload):
        value = value.root
    elif isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        return
    value = cast(dict[str, Any], value)
    rule_evidence = value.get("rule_evidence")
    if isinstance(rule_evidence, dict):
        _collect_missing_leaf_keys(rule_evidence, features)
    for key, item in value.items():
        if key == "rule_evidence":
            continue
        _collect_missing_rule_features(item, features)


def _collect_missing_leaf_keys(
    value: Any, features: set[str], prefix: str = ""
) -> None:
    if isinstance(value, BaseModel):
        _collect_missing_leaf_keys(
            value.model_dump(exclude_none=True), features, prefix
        )
        return
    if isinstance(value, dict):
        value = cast(dict[str, Any], value)
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if child_prefix in _NON_BINDING_MISSING_RULE_FEATURES:
                continue
            _collect_missing_leaf_keys(item, features, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        value = cast(list[Any] | tuple[Any, ...], value)
        for index, item in enumerate(value):
            _collect_missing_leaf_keys(item, features, f"{prefix}[{index}]")
        return
    if _is_missing_rule_value(value):
        features.add(prefix or "unknown")


def _is_missing_rule_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
