from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

from regime_detection.breadth_state import (
    _RISK_RANK as BREADTH_RISK_RANK,
    _data_quality_for_asof as breadth_data_quality_for_asof,
    build_raw_outputs as build_breadth_raw_outputs,
)
from regime_detection.data_quality import assess_series_input_quality, quality_forces_unknown
from regime_detection.event_calendar import (
    _PRECEDENCE,
    _TYPE_TO_LABEL,
    _WINDOWS,
    _month_expiry_date,
    _second_weekday_of_month,
    _sessions_between,
)
from regime_detection.feature_store import FeatureStore
from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.market_context import MarketContext
from regime_detection.models import AxisOutput, BreadthStateOutput, DataQuality, EventCalendarOutput
from regime_detection.trend_character import (
    _RISK_RANK as TREND_CHARACTER_RISK_RANK,
    build_raw_outputs as build_trend_character_raw_outputs,
)
from regime_detection.trend_direction import (
    _RISK_RANK as TREND_DIRECTION_RISK_RANK,
    apply_hysteresis as apply_trend_direction_hysteresis,
    build_raw_outputs as build_trend_direction_raw_outputs,
)
from regime_detection.volatility_state import (
    _RISK_RANK as VOLATILITY_RISK_RANK,
    build_raw_outputs as build_volatility_raw_outputs,
)


@dataclass(frozen=True)
class AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput | BreadthStateOutput]
    stable_labels_by_date: dict[date, str]
    active_labels_by_date: dict[date, str]


@dataclass(frozen=True)
class AxisSeriesBundle:
    trend_direction: AxisSeriesResult
    trend_character: AxisSeriesResult
    volatility_state: AxisSeriesResult
    breadth_state: AxisSeriesResult
    event_calendar: dict[date, EventCalendarOutput]


class AxisSeriesClassifier(Protocol):
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult: ...


class TrendDirectionSeriesClassifier:
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
        close = context.spy_ohlcv["close"]
        features = feature_store.trend_direction
        raw_labels, raw_evidence = build_trend_direction_raw_outputs(features)
        stable_labels, active_labels = apply_trend_direction_hysteresis(
            dates=close.index,
            raw_labels=raw_labels,
            deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
        )
        return _build_axis_outputs(
            dates=close.index.date,
            raw_labels=raw_labels,
            stable_labels=stable_labels,
            active_labels=active_labels,
            raw_evidence=raw_evidence,
            risk_rank=TREND_DIRECTION_RISK_RANK,
            deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
            required_inputs=[close],
            required_trading_days=200,
            max_freshness_days=context.config.data_quality.max_freshness_days,
            min_completeness=context.config.data_quality.min_completeness,
        )


class TrendCharacterSeriesClassifier:
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
        close = context.spy_ohlcv["close"]
        features = feature_store.trend_character
        raw_labels, raw_evidence = build_trend_character_raw_outputs(features)
        stable_labels, active_labels = apply_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=TREND_CHARACTER_RISK_RANK,
            deescalation_days=context.config.hysteresis.trend_character_deescalation_days,
        )
        return _build_axis_outputs(
            dates=close.index.date,
            raw_labels=raw_labels,
            stable_labels=stable_labels,
            active_labels=active_labels,
            raw_evidence=raw_evidence,
            risk_rank=TREND_CHARACTER_RISK_RANK,
            deescalation_days=context.config.hysteresis.trend_character_deescalation_days,
            required_inputs=[close, context.spy_ohlcv["high"], context.spy_ohlcv["low"]],
            required_trading_days=63,
            max_freshness_days=context.config.data_quality.max_freshness_days,
            min_completeness=context.config.data_quality.min_completeness,
        )


class VolatilitySeriesClassifier:
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
        close = context.spy_ohlcv["close"]
        features = feature_store.volatility
        raw_labels, raw_evidence = build_volatility_raw_outputs(features)
        stable_labels, active_labels = apply_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=VOLATILITY_RISK_RANK,
            deescalation_days=context.config.hysteresis.volatility_deescalation_days,
        )
        return _build_axis_outputs(
            dates=close.index.date,
            raw_labels=raw_labels,
            stable_labels=stable_labels,
            active_labels=active_labels,
            raw_evidence=raw_evidence,
            risk_rank=VOLATILITY_RISK_RANK,
            deescalation_days=context.config.hysteresis.volatility_deescalation_days,
            required_inputs=[close],
            required_trading_days=252,
            max_freshness_days=context.config.data_quality.max_freshness_days,
            min_completeness=context.config.data_quality.min_completeness,
        )


class BreadthSeriesClassifier:
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
        spy_close = context.spy_ohlcv["close"]
        rsp_close = context.rsp_close.reindex(context.spy_ohlcv.index)
        features = feature_store.breadth
        raw_labels, raw_evidence = build_breadth_raw_outputs(features)
        stable_labels, active_labels = apply_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=BREADTH_RISK_RANK,
            deescalation_days=context.config.hysteresis.breadth_deescalation_days,
        )
        outputs_by_date: dict[date, BreadthStateOutput] = {}
        stable_by_date: dict[date, str] = {}
        active_by_date: dict[date, str] = {}
        for day, raw, stable, active, evidence in zip(
            spy_close.index.date, raw_labels, stable_labels, active_labels, raw_evidence, strict=True
        ):
            if raw == "unknown":
                output = BreadthStateOutput(
                    mode="etf_proxy",
                    raw_label="unknown",
                    stable_label="unknown",
                    active_label="unknown",
                    evidence={"reason": "insufficient_history", "proxy": "RSP/SPY"},
                    data_quality=DataQuality(
                        status="insufficient_history",
                        freshness_days=None,
                        completeness=None,
                        reason="required_feature_is_nan",
                    ),
                )
            else:
                output = BreadthStateOutput(
                    mode="etf_proxy",
                    raw_label=raw,
                    stable_label=stable,
                    active_label=active,
                    evidence={
                        "proxy": "RSP/SPY",
                        "rule_evidence": evidence,
                        "risk_rank": BREADTH_RISK_RANK,
                        "deescalation_days": context.config.hysteresis.breadth_deescalation_days,
                    },
                    data_quality=breadth_data_quality_for_asof(
                        spy_close=spy_close,
                        rsp_close=rsp_close,
                        as_of_date=day,
                        required_trading_days=50,
                        raw_label=raw,
                        max_freshness_days=context.config.data_quality.max_freshness_days,
                        min_completeness=context.config.data_quality.min_completeness,
                    ),
                )
            outputs_by_date[day] = output
            stable_by_date[day] = output.stable_label
            active_by_date[day] = output.active_label
        return AxisSeriesResult(
            outputs_by_date=outputs_by_date,
            stable_labels_by_date=stable_by_date,
            active_labels_by_date=active_by_date,
        )


def build_axis_series_bundle(*, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesBundle:
    trend_direction = TrendDirectionSeriesClassifier().build(context, feature_store)
    trend_character = TrendCharacterSeriesClassifier().build(context, feature_store)
    volatility_state = VolatilitySeriesClassifier().build(context, feature_store)
    breadth_state = BreadthSeriesClassifier().build(context, feature_store)
    event_calendar = build_event_calendar_series(context)
    return AxisSeriesBundle(
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility_state=volatility_state,
        breadth_state=breadth_state,
        event_calendar=event_calendar,
    )


def build_event_calendar_series(context: MarketContext) -> dict[date, EventCalendarOutput]:
    matches_by_day: dict[date, set[str]] = {day: set() for day in context.sessions}
    if context.normalized_event_calendar is not None and not context.normalized_event_calendar.empty:
        for row in context.normalized_event_calendar.itertuples(index=False):
            label = _TYPE_TO_LABEL.get(str(row.type))
            if label is None:
                continue
            publication_date = row.publication_date
            event_date = row.date
            session_window = list(
                _sessions_between(
                    min(publication_date, event_date) - timedelta(days=20),
                    max(publication_date, event_date) + timedelta(days=20),
                )
            )
            if event_date not in session_window:
                continue
            event_idx = session_window.index(event_date)
            start_offset, end_offset = _WINDOWS[label]
            for idx, day in enumerate(session_window):
                if day not in matches_by_day or day < publication_date:
                    continue
                delta = idx - event_idx
                if start_offset <= delta <= end_offset:
                    matches_by_day[day].add(label)

    expiry_start, expiry_end = context.config.expiry_rules.monthly_options.window_trading_days
    for year, month in sorted({(day.year, day.month) for day in context.sessions}):
        expiry_date = _month_expiry_date(year, month)
        session_window = list(_sessions_between(expiry_date - timedelta(days=10), expiry_date + timedelta(days=10)))
        if expiry_date not in session_window:
            continue
        expiry_idx = session_window.index(expiry_date)
        for idx, day in enumerate(session_window):
            if day not in matches_by_day:
                continue
            delta = idx - expiry_idx
            if expiry_start <= delta <= expiry_end:
                matches_by_day[day].add("expiry_week")

    month_lookup = {
        "second_monday_of_january": 1,
        "second_monday_of_april": 4,
        "second_monday_of_july": 7,
        "second_monday_of_october": 10,
    }
    for year in sorted({day.year for day in context.sessions}):
        for season in context.config.earnings_seasons:
            start = _second_weekday_of_month(
                year=year,
                month=month_lookup[season.start_rule],
                weekday=0,
            )
            end = start + timedelta(days=season.end_offset_days)
            for day in context.sessions:
                if start <= day <= end:
                    matches_by_day[day].add("earnings_season")

    outputs: dict[date, EventCalendarOutput] = {}
    for day in context.sessions:
        ordered = [label for label in _PRECEDENCE if label in matches_by_day[day]]
        selected = ordered[0] if ordered else "normal_calendar"
        outputs[day] = EventCalendarOutput(
            raw_label=selected,
            stable_label=selected,
            active_label=selected,
            evidence={
                "all_matching_events": ordered,
                "selected_via_precedence": selected,
            },
        )
    return outputs


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
    input_by_date = [series for series in required_inputs]
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
