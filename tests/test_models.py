from __future__ import annotations

import json
import ast
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from regime_detection.models import (
    AxisEvidencePayload,
    AxisOutput,
    BreadthStateOutput,
    ChangePointOutput,
    ClusterOutput,
    DataQuality,
    EventCalendarOutput,
    HmmOutput,
    InflationGrowthEvidencePayload,
    InflationGrowthOutput,
    MonetaryPressureEvidencePayload,
    MonetaryPressureV2Output,
    NetworkFragilityEvidencePayload,
    NetworkFragilityOutput,
    RegimeOutput,
    RegimeTimeline,
    StrategyFamilyConstraint,
    StrategyResponse,
    StructuralCausalState,
    TransitionRiskEvidencePayload,
    TransitionRiskOutput,
    VolumeLiquidityEvidencePayload,
    VolumeLiquidityStateOutput,
    _missing_rule_features,
    _project_legacy_v1_transition_risk,
)


def test_legacy_v1_projection_helpers_live_outside_model_boundary() -> None:
    from regime_detection import legacy_v1_wire

    assert _project_legacy_v1_transition_risk is (
        legacy_v1_wire.project_legacy_v1_transition_risk
    )


def test_models_module_is_compatibility_facade_only() -> None:
    tree = ast.parse(Path("src/regime_detection/models.py").read_text())
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    ]

    assert definitions == []


def _data_quality() -> DataQuality:
    return DataQuality(status="ok", freshness_days=0, completeness=1.0)


def _axis(label: str) -> AxisOutput:
    return AxisOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={},
        data_quality=_data_quality(),
    )


def _breadth(label: str) -> BreadthStateOutput:
    return BreadthStateOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={},
        data_quality=_data_quality(),
        mode="etf_proxy",
    )


def _strategy_response() -> StrategyResponse:
    return StrategyResponse(
        position_size_multiplier=1.0,
        allow_trend_following=True,
        allow_mean_reversion=True,
        leverage_allowed=False,
        allow_buy_dip=True,
        allow_breakout=True,
        allow_shorts=False,
        require_confirmation_for_new_longs=False,
        require_confirmation_for_shorts=True,
        log_for_review=False,
        modifiers_applied=[],
    )


def _transition_risk(*, state: str = "stable") -> TransitionRiskOutput:
    return TransitionRiskOutput(
        state=state,
        evidence=TransitionRiskEvidencePayload(
            triggered_rules=[],
            stable_changed_today=False,
            days_since_axis_switch=None,
            axis_switch_count=0,
            recent_axis_switch_count=0,
        ),
        score=0.10,
        score_components={"trend_break": 0.10},
        data_quality=_data_quality(),
    )


def _regime_output(
    *, config_version: str = "test", market: str = "SPY"
) -> RegimeOutput:
    return RegimeOutput(
        engine_version="regime-engine-v-test",
        config_version=config_version,
        as_of_date=date(2023, 12, 14),
        market=market,
        trend_direction=_axis("bull"),
        trend_character=_axis("steady"),
        volatility_state=_axis("normal_vol"),
        breadth_state=_breadth("healthy"),
        structural_causal_state=StructuralCausalState(
            event_calendar=EventCalendarOutput(
                primary_label="normal_calendar",
                matching_labels=("normal_calendar",),
                evidence={},
            ),
        ),
        network_fragility=NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={},
            data_quality=_data_quality(),
        ),
        transition_risk=_transition_risk(),
        strategy_response=_strategy_response(),
    )


def test_axis_evidence_payload_supports_legacy_dict_protocol() -> None:
    left = AxisEvidencePayload(
        rule_evidence={"rule": "trend_above_ma", "value": 1.2},
        hmm_top_state=2,
        hmm_top_state_prob=0.91,
    )
    right = AxisEvidencePayload(
        rule_evidence={"rule": "trend_above_ma", "value": 1.2},
        hmm_top_state=2,
        hmm_top_state_prob=0.91,
    )

    assert "rule_evidence" in left
    assert list(iter(left)) == ["rule_evidence", "hmm_top_state", "hmm_top_state_prob"]
    assert len(left) == 3
    assert dict(left.items())["rule_evidence"]["rule"] == "trend_above_ma"
    assert list(left.keys()) == ["rule_evidence", "hmm_top_state", "hmm_top_state_prob"]
    assert left.get("hmm_top_state") == 2
    assert left == right
    assert left == {
        "rule_evidence": {"rule": "trend_above_ma", "value": 1.2},
        "hmm_top_state": 2,
        "hmm_top_state_prob": 0.91,
    }
    assert AxisEvidencePayload.__eq__(left, object()) is NotImplemented


def test_axis_evidence_payload_rejects_undeclared_business_fields() -> None:
    with pytest.raises(ValidationError):
        AxisEvidencePayload.model_validate({"rule": "misspelled_legacy_key"})


def test_transition_risk_evidence_payload_supports_dict_protocol() -> None:
    left = TransitionRiskEvidencePayload(
        triggered_rules=["trend_break"],
        stable_changed_today=False,
        days_since_axis_switch=None,
        axis_switch_count=1,
        recent_axis_switch_count=1,
    )
    right = TransitionRiskEvidencePayload(
        triggered_rules=["trend_break"],
        stable_changed_today=False,
        days_since_axis_switch=None,
        axis_switch_count=1,
        recent_axis_switch_count=1,
    )

    assert left.get("axis_switch_count") == 1
    assert left.get("missing", "fallback") == "fallback"
    assert left["recent_axis_switch_count"] == 1
    assert "triggered_rules" in left
    assert list(iter(left)) == list(type(left).model_fields)
    assert len(left) == len(type(left).model_fields)
    assert dict(left.items())["triggered_rules"] == ["trend_break"]
    assert list(left.keys()) == list(type(left).model_fields)
    assert list(left.values())[0] == ["trend_break"]
    assert left == right
    assert left == {
        "triggered_rules": ["trend_break"],
        "stable_changed_today": False,
        "days_since_axis_switch": None,
        "axis_switch_count": 1,
        "recent_axis_switch_count": 1,
        "macro_event_labels": [],
    }
    assert TransitionRiskEvidencePayload.__eq__(left, object()) is NotImplemented


def test_v2_axis_outputs_use_typed_axis_specific_evidence_payloads() -> None:
    network = NetworkFragilityOutput(
        raw_label="unknown",
        stable_label="unknown",
        active_label="unknown",
        evidence={
            "rule_evidence": {"rule_path": "unknown_default"},
            "breadth_active_label": "healthy",
            "volatility_active_label": "low_vol",
            "credit_funding_active_label": None,
        },
        data_quality=_data_quality(),
    )
    inflation = InflationGrowthOutput(
        raw_label="macro_neutral",
        stable_label="macro_neutral",
        active_label="macro_neutral",
        evidence={
            "rule_evidence": {"pmi_manufacturing": 50.0},
            "goldilocks_limb_evidence": {"passed_count": 0},
            "credit_funding_active_label": "credit_calm",
            "bias_warning_code": "commodity_proxy_dbc_substitute",
        },
        data_quality=_data_quality(),
    )
    monetary = MonetaryPressureV2Output(
        raw_label="neutral_monetary",
        stable_label="neutral_monetary",
        active_label="neutral_monetary",
        evidence={
            "rule_evidence": {"yield_change_zscore_2y_63d": 0.1},
            "central_bank_text_evidence": {
                "score": 0.0,
                "source": "not_configured",
                "quality": "absent",
            },
        },
        data_quality=_data_quality(),
    )
    volume = VolumeLiquidityStateOutput(
        raw_label="normal_volume",
        stable_label="normal_volume",
        active_label="normal_volume",
        evidence={
            "rule_evidence": {"volume_zscore_20d": 0.5},
            "rule_path": "normal_volume",
            "rule_reason": "volume normal",
        },
        data_quality=_data_quality(),
    )

    assert type(network.evidence) is NetworkFragilityEvidencePayload
    assert network.evidence.breadth_active_label == "healthy"
    assert type(inflation.evidence) is InflationGrowthEvidencePayload
    assert inflation.evidence.credit_funding_active_label == "credit_calm"
    assert type(monetary.evidence) is MonetaryPressureEvidencePayload
    assert monetary.evidence.central_bank_text_evidence is not None
    assert type(volume.evidence) is VolumeLiquidityEvidencePayload
    assert volume.evidence.rule_path == "normal_volume"


def test_v2_typed_evidence_payloads_reject_unknown_business_fields() -> None:
    for payload_cls in (
        NetworkFragilityEvidencePayload,
        InflationGrowthEvidencePayload,
        MonetaryPressureEvidencePayload,
        VolumeLiquidityEvidencePayload,
    ):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            payload_cls.model_validate({"unexpected_field": "silent drift"})


def test_missing_rule_features_handles_none_basemodel_and_list_paths() -> None:
    class NestedRuleEvidence(BaseModel):
        rule_evidence: dict[str, object]

    assert _missing_rule_features(None) == []
    assert _missing_rule_features(
        AxisEvidencePayload(
            rule_evidence={
                "nested": NestedRuleEvidence(
                    rule_evidence={"window": [1.0, None, {"deep": None}]}
                )
            }
        )
    ) == ["nested.rule_evidence.window[1]", "nested.rule_evidence.window[2].deep"]


def test_serializers_omit_none_fields_by_default() -> None:
    cluster = ClusterOutput(
        cluster_id=3,
        distance_to_centroid=1.5,
        model_version="v1",
        mapping_status="map_absent",
        mapping_reason="cluster_label_map_not_configured",
    )
    hmm = HmmOutput(
        top_state=1,
        top_state_prob=0.8,
        n_states=3,
        model_version="v2",
        mapping_status="map_absent",
        mapping_reason="state_label_map_not_configured",
    )
    change_point = ChangePointOutput(score=0.2, method="BOCPD")
    strategy = _strategy_response()
    family = StrategyFamilyConstraint(allowed=True)

    assert cluster.model_dump() == {
        "cluster_id": 3,
        "distance_to_centroid": 1.5,
        "model_version": "v1",
        "mapping_status": "map_absent",
        "mapping_reason": "cluster_label_map_not_configured",
    }
    assert json.loads(cluster.model_dump_json()) == cluster.model_dump()
    assert hmm.model_dump() == {
        "top_state": 1,
        "top_state_prob": 0.8,
        "n_states": 3,
        "model_version": "v2",
        "mapping_status": "map_absent",
        "mapping_reason": "state_label_map_not_configured",
    }
    assert json.loads(hmm.model_dump_json()) == hmm.model_dump()
    assert change_point.model_dump() == {"score": 0.2, "method": "BOCPD"}
    assert json.loads(change_point.model_dump_json()) == change_point.model_dump()
    assert "reason" not in strategy.model_dump()
    assert json.loads(strategy.model_dump_json()) == strategy.model_dump()
    assert family.model_dump() == {"allowed": True}
    assert json.loads(family.model_dump_json()) == family.model_dump()


def test_transition_risk_output_derives_and_preserves_classification_status() -> None:
    insufficient = _transition_risk(state="insufficient_data")
    explicit = TransitionRiskOutput(
        state="stable",
        evidence=TransitionRiskEvidencePayload(
            triggered_rules=[],
            stable_changed_today=False,
            days_since_axis_switch=None,
            axis_switch_count=0,
            recent_axis_switch_count=0,
        ),
        data_quality=_data_quality(),
        classification_status="data_unavailable",
    )

    assert insufficient.classification_status == "insufficient_history"
    assert explicit.classification_status == "data_unavailable"


def test_regime_output_legacy_v1_projection_preserves_archived_wire_shape() -> None:
    output = _regime_output(config_version="core3-v1.0.0")

    payload = output.model_dump()

    assert payload["network_fragility"] == {
        "label": "not_implemented_v1",
        "reason": "breadth_state_used_as_v1_fragility_proxy",
    }
    assert payload["transition_risk"] == {
        "label": "stable",
        "evidence": {
            "triggered_rules": [],
            "stable_changed_today": False,
            "axis_switch_count": 0,
            "recent_axis_switch_count": 0,
            "macro_event_labels": [],
        },
    }
    assert "classification_status" not in payload["trend_direction"]
    assert "classification_status" not in payload["transition_risk"]


def test_regime_output_non_v1_dump_keeps_native_shape_and_json_mode() -> None:
    output = _regime_output(config_version="test", market="SPÝ")

    payload = output.model_dump(mode="json")
    compact = output.model_dump_json()

    assert payload["transition_risk"]["state"] == "stable"
    assert "label" not in payload["transition_risk"]
    assert payload["network_fragility"]["classification_status"] == "no_rule_fired"
    assert compact.startswith('{"engine_version":"regime-engine-v-test"')
    assert "SPÝ" in compact
    assert json.loads(compact) == payload


def test_regime_timeline_dump_and_json_use_legacy_projection_and_compact_format() -> (
    None
):
    output = _regime_output(config_version="core3-v1.0.0", market="SPÝ")
    timeline = RegimeTimeline(
        engine_version="regime-engine-v-test",
        config_version="core3-v1.0.0",
        market="SPÝ",
        start_date=output.as_of_date,
        end_date=output.as_of_date,
        trading_calendar="XNYS",
        outputs=[output],
    )

    payload = timeline.model_dump(mode="json")
    compact = timeline.model_dump_json()
    pretty = timeline.model_dump_json(indent=2)

    assert payload["outputs"][0]["transition_risk"]["label"] == "stable"
    assert compact.startswith('{"engine_version":"regime-engine-v-test"')
    assert "SPÝ" in compact
    assert "\\u00dd" not in compact
    assert "\n  " in pretty
    assert json.loads(compact) == payload
    assert json.loads(pretty) == payload


def test_legacy_transition_risk_projection_leaves_non_dict_payload_unchanged() -> None:
    payload = {"transition_risk": "stable"}

    _project_legacy_v1_transition_risk(payload)

    assert payload == {"transition_risk": "stable"}
