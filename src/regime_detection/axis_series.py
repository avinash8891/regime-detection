from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from regime_detection.axis_builders.breadth import build_breadth_axis_series
from regime_detection.axis_builders.credit_funding import (
    build_credit_funding_axis_series,
    build_credit_funding_proxy_axis_series,
    resolve_credit_funding_effective_output as resolve_credit_funding_effective_output,
    resolve_credit_funding_effective_series,
)
from regime_detection.axis_builders.inflation_growth import build_inflation_growth_axis_series
from regime_detection.axis_builders.monetary_pressure import build_monetary_pressure_axis_series
from regime_detection.axis_builders.network_fragility import build_network_fragility_axis_series
from regime_detection.axis_builders.staleness import (
    _STALENESS_SENTINEL as _STALENESS_SENTINEL,
    _calendar_staleness_days_series as _calendar_staleness_days_series,
    _trading_staleness_series as _trading_staleness_series,
)
from regime_detection.axis_builders.trend_character import build_trend_character_axis_series
from regime_detection.axis_builders.trend_direction import build_trend_direction_axis_series
from regime_detection.axis_builders.volatility import build_volatility_axis_series
from regime_detection.axis_builders.volume_liquidity import build_volume_liquidity_axis_series
from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.event_calendar import compute_event_calendar_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    AxisOutput,
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



def _build_axis_outputs(
    *,
    dates: list[date] | tuple[date, ...],
    raw_labels: list[str],
    stable_labels: list[str],
    active_labels: list[str],
    raw_evidence: list[dict[str, object]],
    risk_rank: dict[str, int],
    deescalation_days: int,
    required_inputs: list,
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
) -> AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput] = {}
    stable_by_date: dict[date, str] = {}
    active_by_date: dict[date, str] = {}
    input_by_date = list(required_inputs)
    for day, raw, stable, active, evidence in zip(
        dates, raw_labels, stable_labels, active_labels, raw_evidence, strict=True
    ):
        dq = assess_series_input_quality(
            as_of_date=day,
            required_inputs=input_by_date,
            required_trading_days=required_trading_days,
            raw_label=raw,
            max_freshness_days=max_freshness_days,
            min_completeness=min_completeness,
        )
        if quality_forces_unknown(dq):
            output = AxisOutput(
                raw_label="unknown",
                stable_label="unknown",
                active_label="unknown",
                evidence={"reason": dq.reason},
                data_quality=dq,
            )
        else:
            output = AxisOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence={
                    "rule_evidence": evidence,
                    "risk_rank": risk_rank,
                    "deescalation_days": deescalation_days,
                },
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
        sessions=context.sessions,
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
