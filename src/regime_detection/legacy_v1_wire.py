from __future__ import annotations

import json
from typing import Any, cast

V1_CONFIG_VERSION = "core3-v1.0.0"


def dump_json_payload(
    payload: dict[str, Any], *, indent: int | None, ensure_ascii: bool
) -> str:
    json_kwargs: dict[str, Any] = {
        "ensure_ascii": ensure_ascii,
    }
    if indent is None:
        json_kwargs["separators"] = (",", ":")
    else:
        json_kwargs["indent"] = indent
    return json.dumps(payload, **json_kwargs)


def project_legacy_v1_wire_shapes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("config_version") != V1_CONFIG_VERSION:
        return payload

    strip_classification_metadata(payload)

    payload["network_fragility"] = {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    project_legacy_v1_transition_risk(payload)
    return payload


def project_legacy_v1_transition_risk(payload: dict[str, Any]) -> None:
    transition_risk = payload.get("transition_risk")
    if not isinstance(transition_risk, dict):
        return

    transition_risk = cast(dict[str, Any], transition_risk)
    label = transition_risk.get("label", transition_risk.get("state"))
    evidence = transition_risk.get("evidence", {})
    payload["transition_risk"] = {"label": label, "evidence": evidence}


def strip_classification_metadata(value: Any) -> None:
    if isinstance(value, dict):
        value = cast(dict[str, Any], value)
        value.pop("classification_status", None)
        value.pop("classification_reason", None)
        value.pop("classification_coverage", None)
        for nested in value.values():
            strip_classification_metadata(nested)
    elif isinstance(value, list):
        value = cast(list[Any], value)
        for item in value:
            strip_classification_metadata(item)
