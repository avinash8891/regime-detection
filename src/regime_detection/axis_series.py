from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from regime_detection.axis_builders.breadth import build_breadth_axis_series
from regime_detection.axis_builders.credit_funding import (
    build_credit_funding_axis_series,
    build_credit_funding_proxy_axis_series,
    resolve_credit_funding_effective_output as resolve_credit_funding_effective_output,
    resolve_credit_funding_effective_series,
)
from regime_detection.axis_builders.inflation_growth import (
    build_inflation_growth_axis_series,
)
from regime_detection.axis_builders.monetary_pressure import (
    build_monetary_pressure_axis_series,
)
from regime_detection.axis_builders.network_fragility import (
    build_network_fragility_axis_series,
)
from regime_detection.axis_builders.staleness import (
    _STALENESS_SENTINEL as _STALENESS_SENTINEL,  # pyright: ignore[reportPrivateUsage]
    _calendar_staleness_days_series as _calendar_staleness_days_series,  # pyright: ignore[reportPrivateUsage]
    _trading_staleness_series as _trading_staleness_series,  # pyright: ignore[reportPrivateUsage]
    staleness_for_source as staleness_for_source,
)
from regime_detection.axis_builders.trend_character import (
    build_trend_character_axis_series,
)
from regime_detection.axis_builders.trend_direction import (
    build_trend_direction_axis_series,
)
from regime_detection.axis_builders.volatility import build_volatility_axis_series
from regime_detection.axis_builders.volume_liquidity import (
    build_volume_liquidity_axis_series,
)
from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.event_calendar import compute_event_calendar_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.hysteresis import apply_data_quality_aware_hysteresis
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    AxisOutput,
    AxisEvidencePayload,
    BreadthStateOutput,
    CreditFundingOutput,
    EventCalendarOutput,
    InflationGrowthOutput,
    MonetaryPressureV2Output,
    NetworkFragilityOutput,
    VolumeLiquidityStateOutput,
)


@dataclass(frozen=True)
class AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput | BreadthStateOutput]
    stable_labels_by_date: dict[date, str]
    active_labels_by_date: dict[date, str]


@dataclass(frozen=True)
class AxisSeriesBundle:
    # TODO(model, owner=regime-maintainers): Consider Pydantic/model validation only after defining the
    # real cross-axis invariants this bundle must enforce. A wrapper-only
    # conversion from dataclass to model would add surface area without safety.
    trend_direction: AxisSeriesResult
    trend_character: AxisSeriesResult
    volatility_state: AxisSeriesResult
    breadth_state: AxisSeriesResult
    event_calendar: dict[date, EventCalendarOutput]
    # V2 §3 network fragility — None in pure-v1 mode (no sector ETF data),
    # populated by build_network_fragility_axis_series when feature_store has
    # the v2 fragility seam.
    network_fragility: dict[date, NetworkFragilityOutput] | None = None
    # V2 §1E volume/liquidity — None in pure-v1 mode (no v2 config),
    # populated by build_volume_liquidity_axis_series when feature_store
    # has the v2 volume_liquidity_v2 seam.
    volume_liquidity_state: dict[date, VolumeLiquidityStateOutput] | None = None
    # V2 §2C credit/funding — None in pure-v1 mode (no v2 config),
    # populated by build_credit_funding_axis_series when feature_store has
    # the credit_funding seam lit.
    credit_funding: dict[date, CreditFundingOutput] | None = None
    # V2 §2C credit/funding PROXY label — None in pure-v1 mode; populated by
    # build_credit_funding_proxy_axis_series on the TLT-vs-HYG/LQD
    # differential. Parallel to `credit_funding`; the effective output below
    # carries the explicit downstream source-selection policy.
    credit_funding_proxy: dict[date, CreditFundingOutput] | None = None
    # V2 §2C effective downstream credit/funding label. OAS and proxy remain
    # visible separately; this map is what cross-axis rules consume.
    credit_funding_effective: dict[date, CreditFundingOutput] | None = None
    # V2 §2A monetary pressure — None in pure-v1 mode (no v2 config), populated
    # by build_monetary_pressure_axis_series when feature_store.monetary is lit
    # AND context.config.monetary_pressure_state is non-None.
    monetary_pressure_state: dict[date, MonetaryPressureV2Output] | None = None
    # V2 §2B inflation/growth — None in pure-v1 mode (no v2 config), populated
    # by build_inflation_growth_axis_series when feature_store.inflation_growth is
    # lit AND context.config.inflation_growth is non-None.
    inflation_growth: dict[date, InflationGrowthOutput] | None = None


@dataclass(frozen=True)
class AxisDependencyContract:
    """Declared payload and failure semantics for one cross-axis edge.

    `AXIS_DEPENDENCIES` remains the build-order graph. This contract is the
    business contract: what crosses the edge and how absence, staleness,
    unknown labels, degraded quality, and invalid sessions are interpreted.
    """

    upstream_axis: str
    downstream_consumer: str
    payload_fields: tuple[str, ...]
    absent_policy: str
    stale_policy: str
    unknown_policy: str
    degraded_policy: str
    invalid_policy: str

    def __post_init__(self) -> None:
        _validate_axis_dependency_policy(
            "absent_policy", self.absent_policy, _ABSENT_POLICIES
        )
        _validate_axis_dependency_policy(
            "stale_policy", self.stale_policy, _STALE_POLICIES
        )
        _validate_axis_dependency_policy(
            "unknown_policy", self.unknown_policy, _UNKNOWN_POLICIES
        )
        _validate_axis_dependency_policy(
            "degraded_policy", self.degraded_policy, _DEGRADED_POLICIES
        )
        _validate_axis_dependency_policy(
            "invalid_policy", self.invalid_policy, _INVALID_POLICIES
        )


_ABSENT_POLICIES = frozenset(
    {
        "fallback_unknown_when_omitted",
        "omit_component",
        "pass_none",
        "pass_none_to_falsify_predicate",
        "raise_on_missing_required_axis",
        "timeline_placeholder_unknown",
    }
)
_STALE_POLICIES = frozenset(
    {
        "label_only_data_quality_not_visible",
        "not_encoded_on_payload",
        "omit_if_not_classified",
    }
)
_UNKNOWN_POLICIES = frozenset(
    {
        "force_transition_score_missing_axis_data_quality",
        "omit_if_not_classified",
        "pass_empty_or_matching_labels",
        "pass_state",
        "pass_unknown_label",
    }
)
_DEGRADED_POLICIES = frozenset(
    {
        "label_only_data_quality_not_visible",
        "not_encoded_on_payload",
        "omit_if_not_classified",
    }
)
_INVALID_POLICIES = frozenset(
    {
        "omit_missing_session",
        "raise_on_missing_output",
        "raise_on_missing_session",
        "raise_on_missing_session_when_present",
        "raise_when_configured_but_unavailable",
    }
)


def _validate_axis_dependency_policy(
    field_name: str, value: str, allowed_values: frozenset[str]
) -> None:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(
            f"{field_name} {value!r} is not declared in AxisDependencyContract "
            f"policy vocabulary: {allowed}"
        )


AXIS_BUILD_ORDER: tuple[str, ...] = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "event_calendar",
    "credit_funding",
    "credit_funding_proxy",
    "credit_funding_effective",
    "network_fragility",
    "volume_liquidity_state",
    "monetary_pressure_state",
    "inflation_growth",
)

AXIS_DEPENDENCY_CONTRACTS: tuple[AxisDependencyContract, ...] = (
    # These edges intentionally declare the current label-only contract. If a
    # downstream consumer starts needing upstream evidence, stable_label, or
    # data_quality, this table must change before the wire shape changes.
    AxisDependencyContract(
        upstream_axis="breadth_state",
        downstream_consumer="network_fragility",
        payload_fields=("active_label",),
        absent_policy="fallback_unknown_when_omitted",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session_when_present",
    ),
    AxisDependencyContract(
        upstream_axis="volatility_state",
        downstream_consumer="network_fragility",
        payload_fields=("active_label",),
        absent_policy="fallback_unknown_when_omitted",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session_when_present",
    ),
    AxisDependencyContract(
        upstream_axis="credit_funding_effective",
        downstream_consumer="network_fragility",
        payload_fields=("active_label",),
        absent_policy="pass_none_to_falsify_predicate",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session_when_present",
    ),
    AxisDependencyContract(
        upstream_axis="credit_funding_effective",
        downstream_consumer="inflation_growth",
        payload_fields=("active_label",),
        absent_policy="pass_none_to_falsify_predicate",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session_when_present",
    ),
    AxisDependencyContract(
        upstream_axis="trend_direction",
        downstream_consumer="transition_risk_history",
        payload_fields=("stable_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="trend_character",
        downstream_consumer="transition_risk_history",
        payload_fields=("stable_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="volatility_state",
        downstream_consumer="transition_risk_history",
        payload_fields=("stable_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="breadth_state",
        downstream_consumer="transition_risk_history",
        payload_fields=("stable_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="trend_direction",
        downstream_consumer="transition_risk_selection",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="force_transition_score_missing_axis_data_quality",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="trend_character",
        downstream_consumer="transition_risk_selection",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="force_transition_score_missing_axis_data_quality",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="volatility_state",
        downstream_consumer="transition_risk_selection",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="force_transition_score_missing_axis_data_quality",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="breadth_state",
        downstream_consumer="transition_risk_selection",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="force_transition_score_missing_axis_data_quality",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="event_calendar",
        downstream_consumer="transition_score",
        payload_fields=("matching_labels",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="not_encoded_on_payload",
        unknown_policy="pass_empty_or_matching_labels",
        degraded_policy="not_encoded_on_payload",
        invalid_policy="raise_on_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="credit_funding_effective",
        downstream_consumer="transition_score",
        payload_fields=("active_label", "classification_status"),
        absent_policy="omit_component",
        stale_policy="omit_if_not_classified",
        unknown_policy="omit_if_not_classified",
        degraded_policy="omit_if_not_classified",
        invalid_policy="omit_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="volume_liquidity_state",
        downstream_consumer="transition_score",
        payload_fields=("active_label", "classification_status"),
        absent_policy="omit_component",
        stale_policy="omit_if_not_classified",
        unknown_policy="omit_if_not_classified",
        degraded_policy="omit_if_not_classified",
        invalid_policy="omit_missing_session",
    ),
    AxisDependencyContract(
        upstream_axis="trend_direction",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="trend_character",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="volatility_state",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="breadth_state",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="network_fragility",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="timeline_placeholder_unknown",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="monetary_pressure_state",
        downstream_consumer="cohort_routing",
        payload_fields=("active_label",),
        absent_policy="pass_none",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_when_configured_but_unavailable",
    ),
    AxisDependencyContract(
        upstream_axis="trend_direction",
        downstream_consumer="strategy_response",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="trend_character",
        downstream_consumer="strategy_response",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="volatility_state",
        downstream_consumer="strategy_response",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="breadth_state",
        downstream_consumer="strategy_response",
        payload_fields=("active_label",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="label_only_data_quality_not_visible",
        unknown_policy="pass_unknown_label",
        degraded_policy="label_only_data_quality_not_visible",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="transition_risk",
        downstream_consumer="strategy_response",
        payload_fields=("state",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="not_encoded_on_payload",
        unknown_policy="pass_state",
        degraded_policy="not_encoded_on_payload",
        invalid_policy="raise_on_missing_output",
    ),
    AxisDependencyContract(
        upstream_axis="event_calendar",
        downstream_consumer="strategy_response",
        payload_fields=("matching_labels",),
        absent_policy="raise_on_missing_required_axis",
        stale_policy="not_encoded_on_payload",
        unknown_policy="pass_empty_or_matching_labels",
        degraded_policy="not_encoded_on_payload",
        invalid_policy="raise_on_missing_output",
    ),
)


def _dependency_contracts_by_downstream(
    contracts: tuple[AxisDependencyContract, ...],
) -> dict[str, tuple[AxisDependencyContract, ...]]:
    out: dict[str, list[AxisDependencyContract]] = {}
    for contract in contracts:
        out.setdefault(contract.downstream_consumer, []).append(contract)
    return {
        downstream: tuple(downstream_contracts)
        for downstream, downstream_contracts in out.items()
    }


_AXIS_BUILD_DEPENDENCY_CONSUMERS = {
    "network_fragility",
    "inflation_growth",
}

AXIS_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    downstream: tuple(contract.upstream_axis for contract in contracts)
    for downstream, contracts in _dependency_contracts_by_downstream(
        AXIS_DEPENDENCY_CONTRACTS
    ).items()
    if downstream in _AXIS_BUILD_DEPENDENCY_CONSUMERS
}


def _validate_axis_dependency_order(
    build_order: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
) -> None:
    positions = {name: idx for idx, name in enumerate(build_order)}
    contracted_build_dependencies = {
        downstream: tuple(contract.upstream_axis for contract in contracts)
        for downstream, contracts in _dependency_contracts_by_downstream(
            AXIS_DEPENDENCY_CONTRACTS
        ).items()
        if downstream in positions
    }
    for axis, required_axes in contracted_build_dependencies.items():
        if axis not in dependencies:
            raise ValueError(f"dependency graph omits contracted axis {axis!r}")
        if dependencies[axis] != required_axes:
            raise ValueError(
                f"dependency graph for {axis!r} does not match declared contracts"
            )
    for axis, required_axes in dependencies.items():
        if axis not in positions:
            raise ValueError(f"axis build order missing declared axis {axis!r}")
        for required in required_axes:
            if required not in positions:
                raise ValueError(
                    f"axis build order dependency {axis!r}->{required!r} is undeclared"
                )
            if positions[required] >= positions[axis]:
                raise ValueError(
                    f"axis {axis!r} must build after dependency {required!r}"
                )


def build_event_calendar_series(
    context: MarketContext,
) -> dict[date, EventCalendarOutput]:
    """Bulk wrapper around :func:`compute_event_calendar_outputs` — pulls
    ``sessions`` and the pre-normalized event calendar off the context."""
    return compute_event_calendar_outputs(
        sessions=context.sessions,
        normalized_event_calendar=context.normalized_event_calendar,
        config=context.config,
    )


def _build_axis_outputs(  # pyright: ignore[reportUnusedFunction]
    *,
    dates: list[date] | tuple[date, ...],
    raw_labels: list[str],
    raw_evidence: list[dict[str, object]],
    risk_rank: dict[str, int],
    deescalation_days_by_label: dict[str, int],
    default_deescalation_days: int,
    max_unknown_freeze_days: int,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
) -> AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput] = {}
    stable_by_date: dict[date, str] = {}
    active_by_date: dict[date, str] = {}
    input_by_date = list(required_inputs)
    data_quality = [
        assess_series_input_quality(
            as_of_date=day,
            required_inputs=input_by_date,
            required_trading_days=required_trading_days,
            raw_label=raw,
            max_freshness_days=max_freshness_days,
            min_completeness=min_completeness,
        )
        for day, raw in zip(dates, raw_labels, strict=True)
    ]
    stable_labels, active_labels, frozen_labels = apply_data_quality_aware_hysteresis(
        raw_labels=raw_labels,
        risk_rank=risk_rank,
        deescalation_days_by_label=deescalation_days_by_label,
        data_quality=data_quality,
        default_deescalation_days=default_deescalation_days,
        max_unknown_freeze_days=max_unknown_freeze_days,
    )
    for day, raw, stable, active, evidence, dq, is_frozen in zip(
        dates,
        raw_labels,
        stable_labels,
        active_labels,
        raw_evidence,
        data_quality,
        frozen_labels,
        strict=True,
    ):
        if quality_forces_unknown(dq):
            output = AxisOutput(
                raw_label="unknown",
                stable_label=stable if is_frozen else "unknown",
                active_label=active if is_frozen else "unknown",
                evidence=AxisEvidencePayload(
                    reason=dq.reason,
                    data_quality_freeze=True if is_frozen else None,
                ),
                data_quality=dq,
            )
        else:
            output = AxisOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence=AxisEvidencePayload(
                    rule_evidence=evidence,
                    risk_rank=risk_rank,
                    deescalation_days=default_deescalation_days,
                ),
                data_quality=dq,
            )
        outputs_by_date[day] = output
        stable_by_date[day] = output.stable_label
        active_by_date[day] = output.active_label
    return AxisSeriesResult(
        outputs_by_date=outputs_by_date,
        stable_labels_by_date=stable_by_date,
        active_labels_by_date=active_by_date,
    )


def build_axis_series_bundle(
    *, context: MarketContext, feature_store: FeatureStore
) -> AxisSeriesBundle:
    _validate_axis_dependency_order(AXIS_BUILD_ORDER, AXIS_DEPENDENCIES)
    trend_direction = build_trend_direction_axis_series(context, feature_store)
    trend_character = build_trend_character_axis_series(context, feature_store)
    volatility_state = build_volatility_axis_series(context, feature_store)
    breadth_state = build_breadth_axis_series(context, feature_store)
    event_calendar = build_event_calendar_series(context)
    credit_funding = build_credit_funding_axis_series(context, feature_store)
    credit_funding_proxy = build_credit_funding_proxy_axis_series(
        context, feature_store
    )
    credit_funding_effective = resolve_credit_funding_effective_series(
        sessions=list(context.sessions),
        oas_by_date=credit_funding,
        proxy_by_date=credit_funding_proxy,
    )
    network_fragility = build_network_fragility_axis_series(
        context,
        feature_store,
        breadth_active_labels_by_date=breadth_state.active_labels_by_date,
        volatility_active_labels_by_date=volatility_state.active_labels_by_date,
        # Downstream rules consume the effective credit state, not raw OAS.
        # Raw OAS can be stale before the OAS history starts while TLT proxy
        # fallback is classified and explicitly recorded in the effective seam.
        credit_funding_active_labels_by_date=(
            {day: out.active_label for day, out in credit_funding_effective.items()}
            if credit_funding_effective is not None
            else None
        ),
    )
    volume_liquidity_state = build_volume_liquidity_axis_series(context, feature_store)
    monetary_pressure_state = build_monetary_pressure_axis_series(
        context, feature_store
    )
    inflation_growth = build_inflation_growth_axis_series(
        context,
        feature_store,
        # Keep inflation/growth aligned with network fragility: credit is stale
        # only when the effective OAS/proxy selection cannot classify the date.
        credit_funding_active_labels_by_date=(
            {day: out.active_label for day, out in credit_funding_effective.items()}
            if credit_funding_effective is not None
            else None
        ),
    )
    return AxisSeriesBundle(
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility_state=volatility_state,
        breadth_state=breadth_state,
        event_calendar=event_calendar,
        network_fragility=network_fragility,
        volume_liquidity_state=volume_liquidity_state,
        credit_funding=credit_funding,
        credit_funding_proxy=credit_funding_proxy,
        credit_funding_effective=credit_funding_effective,
        monetary_pressure_state=monetary_pressure_state,
        inflation_growth=inflation_growth,
    )
