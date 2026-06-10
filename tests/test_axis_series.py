from __future__ import annotations

from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.axis_series import (
    AXIS_BUILD_ORDER,
    AXIS_DEPENDENCIES,
    AXIS_DEPENDENCY_CONTRACTS,
    AxisDependencyContract,
    AxisSeriesBundle,
    _build_axis_outputs,
    _dependency_contracts_by_downstream,
    _validate_axis_dependency_order,
    build_axis_series_bundle,
)
from regime_detection.config import load_regime_config
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V1_CONFIG_PATH = (
    _REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
)


def test_axis_series_bundle_contract_names_every_timeline_axis() -> None:
    bundle_fields = {field.name for field in fields(AxisSeriesBundle)}

    assert bundle_fields == {
        "trend_direction",
        "trend_character",
        "volatility_state",
        "breadth_state",
        "event_calendar",
        "network_fragility",
        "volume_liquidity_state",
        "credit_funding",
        "credit_funding_proxy",
        "credit_funding_effective",
        "monetary_pressure_state",
        "inflation_growth",
    }


def test_axis_build_order_satisfies_declared_dependencies() -> None:
    assert AXIS_DEPENDENCIES["network_fragility"] == (
        "breadth_state",
        "volatility_state",
        "credit_funding_effective",
    )
    assert AXIS_DEPENDENCIES["inflation_growth"] == ("credit_funding_effective",)
    _validate_axis_dependency_order(AXIS_BUILD_ORDER, AXIS_DEPENDENCIES)


def test_axis_dependency_contracts_declare_every_current_cross_axis_edge() -> None:
    expected_edges = {
        ("breadth_state", "network_fragility"),
        ("volatility_state", "network_fragility"),
        ("credit_funding_effective", "network_fragility"),
        ("credit_funding_effective", "inflation_growth"),
        ("trend_direction", "transition_risk_history"),
        ("trend_character", "transition_risk_history"),
        ("volatility_state", "transition_risk_history"),
        ("breadth_state", "transition_risk_history"),
        ("trend_direction", "transition_risk_selection"),
        ("trend_character", "transition_risk_selection"),
        ("volatility_state", "transition_risk_selection"),
        ("breadth_state", "transition_risk_selection"),
        ("event_calendar", "transition_score"),
        ("credit_funding_effective", "transition_score"),
        ("volume_liquidity_state", "transition_score"),
        ("trend_direction", "cohort_routing"),
        ("trend_character", "cohort_routing"),
        ("volatility_state", "cohort_routing"),
        ("breadth_state", "cohort_routing"),
        ("network_fragility", "cohort_routing"),
        ("monetary_pressure_state", "cohort_routing"),
        ("trend_direction", "strategy_response"),
        ("trend_character", "strategy_response"),
        ("volatility_state", "strategy_response"),
        ("breadth_state", "strategy_response"),
        ("transition_risk", "strategy_response"),
        ("event_calendar", "strategy_response"),
    }

    assert {
        (contract.upstream_axis, contract.downstream_consumer)
        for contract in AXIS_DEPENDENCY_CONTRACTS
    } == expected_edges


def test_axis_dependency_contracts_name_payload_and_failure_semantics() -> None:
    for contract in AXIS_DEPENDENCY_CONTRACTS:
        assert isinstance(contract, AxisDependencyContract)
        assert contract.payload_fields
        assert contract.absent_policy
        assert contract.stale_policy
        assert contract.unknown_policy
        assert contract.degraded_policy
        assert contract.invalid_policy


def test_axis_dependency_contract_rejects_unknown_policy_strings() -> None:
    with pytest.raises(ValueError, match="absent_policy"):
        AxisDependencyContract(
            upstream_axis="breadth_state",
            downstream_consumer="network_fragility",
            payload_fields=("active_label",),
            absent_policy="typo_unknown_policy",
            stale_policy="label_only_data_quality_not_visible",
            unknown_policy="pass_unknown_label",
            degraded_policy="label_only_data_quality_not_visible",
            invalid_policy="raise_on_missing_session_when_present",
        )


def test_axis_dependency_order_rejects_missing_build_graph_contract_coverage() -> None:
    incomplete_dependencies = {
        "network_fragility": (
            "breadth_state",
            "volatility_state",
            "credit_funding_effective",
        )
    }

    with pytest.raises(ValueError, match="dependency graph omits contracted axis"):
        _validate_axis_dependency_order(AXIS_BUILD_ORDER, incomplete_dependencies)


def test_declared_axis_dependencies_are_derived_from_contract_graph() -> None:
    contracts_by_downstream = _dependency_contracts_by_downstream(
        AXIS_DEPENDENCY_CONTRACTS
    )

    assert AXIS_DEPENDENCIES == {
        "network_fragility": (
            "breadth_state",
            "volatility_state",
            "credit_funding_effective",
        ),
        "inflation_growth": ("credit_funding_effective",),
    }
    assert (
        tuple(
            contract.upstream_axis
            for contract in contracts_by_downstream["network_fragility"]
        )
        == AXIS_DEPENDENCIES["network_fragility"]
    )


def test_contract_graph_captures_label_only_credit_edge_semantics() -> None:
    contracts = {
        (contract.upstream_axis, contract.downstream_consumer): contract
        for contract in AXIS_DEPENDENCY_CONTRACTS
    }

    credit_to_network = contracts[("credit_funding_effective", "network_fragility")]
    assert credit_to_network.payload_fields == ("active_label",)
    assert credit_to_network.absent_policy == "pass_none_to_falsify_predicate"
    assert credit_to_network.stale_policy == "label_only_data_quality_not_visible"
    assert credit_to_network.invalid_policy == "raise_on_missing_session_when_present"

    credit_to_score = contracts[("credit_funding_effective", "transition_score")]
    assert credit_to_score.absent_policy == "omit_component"
    assert credit_to_score.unknown_policy == "omit_if_not_classified"


def test_build_axis_series_bundle_emits_session_keyed_outputs_for_core_axes(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=load_regime_config(_V1_CONFIG_PATH),
        event_calendar=event_calendar_df,
    )
    feature_store = build_feature_store(context)

    bundle = build_axis_series_bundle(context=context, feature_store=feature_store)
    expected_dates = set(context.sessions)

    assert set(bundle.trend_direction.outputs_by_date) == expected_dates
    assert set(bundle.trend_character.outputs_by_date) == expected_dates
    assert set(bundle.volatility_state.outputs_by_date) == expected_dates
    assert set(bundle.breadth_state.outputs_by_date) == expected_dates
    assert set(bundle.event_calendar) == expected_dates


def test_core_axis_output_builder_holds_dq_gap_by_departed_label_threshold() -> None:
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]
    index = pd.DatetimeIndex(dates)
    required = pd.Series([1.0, None, None, None], index=index)

    result = _build_axis_outputs(
        dates=dates,
        raw_labels=["bear", "unknown", "unknown", "unknown"],
        raw_evidence=[{}, {}, {}, {}],
        risk_rank={"unknown": 2, "bull": 0, "bear": 3},
        deescalation_days_by_label={"bear": 5, "unknown": 0},
        default_deescalation_days=0,
        max_unknown_freeze_days=2,
        required_inputs=[required],
        required_trading_days=1,
        max_freshness_days=0,
        min_completeness=1.0,
    )

    assert result.outputs_by_date[dates[1]].raw_label == "unknown"
    assert result.outputs_by_date[dates[1]].stable_label == "bear"
    assert result.outputs_by_date[dates[1]].active_label == "bear"
    assert result.outputs_by_date[dates[1]].evidence["data_quality_freeze"] is True
    assert result.outputs_by_date[dates[2]].stable_label == "bear"
    assert result.outputs_by_date[dates[3]].stable_label == "bear"
