from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from regime_detection._axis_result import AxisSeriesResult
from regime_detection.event_calendar import compute_event_calendar_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    CreditFundingOutput,
    EventCalendarOutput,
    InflationGrowthOutput,
    MonetaryPressureV2Output,
    NetworkFragilityOutput,
    VolumeLiquidityStateOutput,
)
import regime_detection.breadth_state as _breadth_state_mod
import regime_detection.credit_funding as _credit_funding_mod
import regime_detection.inflation_growth as _inflation_growth_mod
import regime_detection.monetary_pressure as _monetary_pressure_mod
import regime_detection.network_fragility_rules as _network_fragility_mod
import regime_detection.trend_character as _trend_character_mod
import regime_detection.trend_direction as _trend_direction_mod
import regime_detection.volatility_state as _volatility_state_mod
import regime_detection.volume_liquidity_rules as _volume_liquidity_mod

# Lookback windows for data-quality sufficiency gates (cite the spec section per window).
_TREND_DIRECTION_MIN_SESSIONS = 200   # §1A: 200-day SMA lookback
_TREND_CHARACTER_MIN_SESSIONS = 63    # §1B: 63-day drawdown lookback
_VOLATILITY_MIN_SESSIONS = 252        # §1C: 252-day realized-vol window
_BREADTH_MIN_SESSIONS = 50            # §1D: breadth 50-period minimum


@dataclass(frozen=True)
class AxisSeriesBundle:
    # TODO(model): Consider Pydantic/model validation only after defining the
    # real cross-axis invariants this bundle must enforce. A wrapper-only
    # conversion from dataclass to model would add surface area without safety.
    trend_direction: AxisSeriesResult
    trend_character: AxisSeriesResult
    volatility_state: AxisSeriesResult
    breadth_state: AxisSeriesResult
    event_calendar: dict[date, EventCalendarOutput]
    # V2 §3 network fragility — None in pure-v1 mode (no sector ETF data),
    # populated by NetworkFragilitySeriesClassifier when feature_store has
    # the v2 fragility seam. Slice 1 fills in the real classifier rules.
    network_fragility: dict[date, NetworkFragilityOutput] | None = None
    # V2 §1E volume/liquidity — None in pure-v1 mode (no v2 config),
    # populated by VolumeLiquidityStateSeriesClassifier when feature_store
    # has the v2 volume_liquidity_v2 seam (Slice 2.7).
    volume_liquidity_state: dict[date, VolumeLiquidityStateOutput] | None = None
    # V2 §2C credit/funding — None in pure-v1 mode (no v2 config),
    # populated by CreditFundingSeriesClassifier when feature_store has
    # the credit_funding seam lit (Slice 4).
    credit_funding: dict[date, CreditFundingOutput] | None = None
    # V2 §2C credit/funding PROXY label — None in pure-v1 mode; populated by
    # CreditFundingSeriesClassifier.build_proxy on the TLT-vs-HYG/LQD
    # differential. Parallel to `credit_funding`, never blended (Log #71).
    credit_funding_proxy: dict[date, CreditFundingOutput] | None = None
    # V2 §2A monetary pressure — None in pure-v1 mode (no v2 config), populated
    # by MonetaryPressureV2SeriesClassifier when feature_store.monetary is lit
    # AND context.config.monetary_pressure_state is non-None (Ambiguity Log #46).
    monetary_pressure_state: dict[date, MonetaryPressureV2Output] | None = None
    # V2 §2B inflation/growth — None in pure-v1 mode (no v2 config), populated
    # by InflationGrowthSeriesClassifier when feature_store.inflation_growth is
    # lit AND context.config.inflation_growth is non-None (Slice 5).
    inflation_growth: dict[date, InflationGrowthOutput] | None = None


def build_axis_series_bundle(*, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesBundle:
    trend_direction = _trend_direction_mod.build_axis_series(
        context, feature_store, required_trading_days=_TREND_DIRECTION_MIN_SESSIONS
    )
    trend_character = _trend_character_mod.build_axis_series(
        context, feature_store, required_trading_days=_TREND_CHARACTER_MIN_SESSIONS
    )
    volatility_state = _volatility_state_mod.build_axis_series(
        context, feature_store, required_trading_days=_VOLATILITY_MIN_SESSIONS
    )
    breadth_state = _breadth_state_mod.build_axis_series(
        context, feature_store, required_trading_days=_BREADTH_MIN_SESSIONS
    )
    event_calendar = build_event_calendar_series(context)
    credit_funding = _credit_funding_mod.build_axis_series(context, feature_store)
    credit_funding_proxy = _credit_funding_mod.build_axis_series_proxy(context, feature_store)
    network_fragility = _network_fragility_mod.build_axis_series(
        context,
        feature_store,
        breadth_active_labels_by_date=breadth_state.active_labels_by_date,
        volatility_active_labels_by_date=volatility_state.active_labels_by_date,
        credit_funding_active_labels_by_date=(
            {day: out.active_label for day, out in credit_funding.items()}
            if credit_funding is not None
            else None
        ),
    )
    volume_liquidity_state = _volume_liquidity_mod.build_axis_series(context, feature_store)
    monetary_pressure_state = _monetary_pressure_mod.build_axis_series(context, feature_store)
    inflation_growth = _inflation_growth_mod.build_axis_series(
        context,
        feature_store,
        credit_funding_active_labels_by_date=(
            {day: out.active_label for day, out in credit_funding.items()}
            if credit_funding is not None
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
        monetary_pressure_state=monetary_pressure_state,
        inflation_growth=inflation_growth,
    )


def build_event_calendar_series(context: MarketContext) -> dict[date, EventCalendarOutput]:
    """Bulk wrapper around :func:`compute_event_calendar_outputs` — pulls
    ``sessions`` and the pre-normalized event calendar off the context."""
    return compute_event_calendar_outputs(
        sessions=context.sessions,
        normalized_event_calendar=context.normalized_event_calendar,
        config=context.config,
    )


