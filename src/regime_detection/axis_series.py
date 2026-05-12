from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

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
from regime_detection.hysteresis import (
    apply_asymmetric_hysteresis,
    apply_per_label_asymmetric_hysteresis,
)
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    DataQuality,
    EventCalendarOutput,
    NetworkFragilityOutput,
)
from regime_detection.network_fragility_rules import (
    NETWORK_FRAGILITY_RISK_RANK,
    NetworkFragilityLabel,
    build_rule_inputs_for_date,
    evaluate_rules,
)
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
    # V2 §3 network fragility — None in pure-v1 mode (no sector ETF data),
    # populated by NetworkFragilitySeriesClassifier when feature_store has
    # the v2 fragility seam. Slice 1 fills in the real classifier rules.
    network_fragility: dict[date, NetworkFragilityOutput] | None = None


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


class NetworkFragilitySeriesClassifier:
    """V2 §3 network fragility classifier — full pipeline (Slice 1.4).

    Pipeline per v2 spec §3.2–§3.7:

      1. Read pre-computed features from ``feature_store.network_fragility``
         (compute_features → v2 §3.2). If the seam is None (no sector ETF
         data) return None so the timeline falls back to the v2 "unknown"
         placeholder shape.
      2. For each session date, materialize the per-day scalar rule inputs
         (build_rule_inputs_for_date) and assess data quality
         (assess_series_input_quality + quality_forces_unknown). Quality
         failures override the rule output with "unknown".
      3. Cross-reference V1 axes (breadth_state.active_label,
         volatility_state.active_label) for that date.
      4. Evaluate rules (evaluate_rules) → raw label per v2 §3.4 precedence.
         credit_funding_label is hard-coded to None until Slice 4 ships the
         v2 §2C axis; per network_fragility_rules.evaluate_systemic_stress
         this short-circuits the systemic_stress rule and precedence falls
         through to correlation_to_one.
      5. Apply per-label asymmetric hysteresis (v2 §3.7) over the raw label
         series.
      6. Emit one NetworkFragilityOutput per session.
    """

    def build(
        self,
        context: MarketContext,
        feature_store: FeatureStore,
        breadth_active_labels_by_date: dict[date, str] | None = None,
        volatility_active_labels_by_date: dict[date, str] | None = None,
    ) -> dict[date, NetworkFragilityOutput] | None:
        features = feature_store.network_fragility
        if features is None:
            return None

        network_fragility_config = context.config.network_fragility
        if network_fragility_config is None:
            # Defensive: the feature seam is populated but config is missing.
            # Treat as no v2 axis available rather than crashing the engine.
            return None

        spy_close = context.spy_ohlcv["close"]
        volatility_features = feature_store.volatility
        realized_vol_pct = volatility_features.realized_vol_percentile_252d
        vix_pct = (
            volatility_features.vix_percentile_252d
            if volatility_features.vix_percentile_252d is not None
            else pd.Series(float("nan"), index=spy_close.index)
        )

        # Required-input check (v2 §2.8 data quality).
        # The percentile features are derived and are structurally NaN until
        # the 504d / 252d windows fill — the rules themselves return False on
        # NaN inputs (falling through to "unknown" per v2 §3.3), so we gate
        # on the underlying primary inputs only and let day-level NaN
        # propagate through the rule engine.
        required_inputs: list[pd.Series] = [
            features.avg_pairwise_corr_63d,
            features.largest_eigenvalue_share,
            features.effective_rank,
            features.dispersion_ratio,
            spy_close,
        ]
        # The 63d correlation window is the longest mandatory raw-input
        # lookback (v2 §3.2 line 554-558). The 504d percentile / 21d slope
        # NaN-ness is handled inside the rule predicates per spec.
        required_trading_days = network_fragility_config.correlation_lookback_days
        max_freshness_days = context.config.data_quality.max_freshness_days
        min_completeness = context.config.data_quality.min_completeness

        raw_labels: list[NetworkFragilityLabel] = []
        per_day_data_quality: list[DataQuality] = []
        per_day_evidence: list[dict[str, object]] = []

        for day in context.sessions:
            dt = pd.Timestamp(day)

            # Defensive: the volatility series may not include early sessions
            # if cold-start clipping diverges. Reindex on the fly.
            day_quality = assess_series_input_quality(
                as_of_date=day,
                required_inputs=required_inputs,
                required_trading_days=required_trading_days,
                # Pre-quality: pass a placeholder so the helper's
                # "raw_label == 'unknown'" branch doesn't fire here — we
                # compute the real raw label below and re-check via
                # quality_forces_unknown.
                raw_label="placeholder",
                max_freshness_days=max_freshness_days,
                min_completeness=min_completeness,
            )

            if quality_forces_unknown(day_quality):
                raw_labels.append("unknown")
                per_day_data_quality.append(day_quality)
                per_day_evidence.append({"reason": day_quality.reason or "insufficient_data"})
                continue

            rule_inputs = build_rule_inputs_for_date(
                features=features,
                dt=dt,
                spy_close=spy_close,
                realized_vol_percentile_252d=realized_vol_pct,
                vix_percentile_252d=vix_pct,
            )

            breadth_label = "unknown"
            if breadth_active_labels_by_date is not None:
                breadth_label = breadth_active_labels_by_date.get(day, "unknown")
            volatility_label = "unknown"
            if volatility_active_labels_by_date is not None:
                volatility_label = volatility_active_labels_by_date.get(day, "unknown")

            label = evaluate_rules(
                inputs=rule_inputs,
                config=network_fragility_config.rules,
                breadth_label=breadth_label,  # type: ignore[arg-type]
                volatility_label=volatility_label,  # type: ignore[arg-type]
                # Slice 4 (v2 §2C credit/funding) is not yet implemented.
                # network_fragility_rules.evaluate_systemic_stress short-
                # circuits to False on None and precedence falls through
                # to correlation_to_one per v2 §3.4.
                credit_funding_label=None,
            )
            raw_labels.append(label)
            per_day_data_quality.append(day_quality)
            per_day_evidence.append({
                "rule_evidence": {
                    "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
                    "largest_eigenvalue_share_percentile_504d": rule_inputs.largest_eigenvalue_share_percentile_504d,
                    "effective_rank_percentile_504d": rule_inputs.effective_rank_percentile_504d,
                    "dispersion_ratio_percentile_252d": rule_inputs.dispersion_ratio_percentile_252d,
                    "realized_vol_percentile_252d": rule_inputs.realized_vol_percentile_252d,
                    "drawdown_21d": rule_inputs.drawdown_21d,
                    "vix_percentile_252d": rule_inputs.vix_percentile_252d,
                },
                "breadth_active_label": breadth_label,
                "volatility_active_label": volatility_label,
                "credit_funding_active_label": None,
            })

        stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=NETWORK_FRAGILITY_RISK_RANK,
            deescalation_days_by_label=network_fragility_config.deescalation_days_by_label,
        )

        outputs: dict[date, NetworkFragilityOutput] = {}
        for day, raw, stable, active, dq, evidence in zip(
            context.sessions,
            raw_labels,
            stable_labels,
            active_labels,
            per_day_data_quality,
            per_day_evidence,
            strict=True,
        ):
            outputs[day] = NetworkFragilityOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence=evidence,
                data_quality=dq,
            )
        return outputs


def build_axis_series_bundle(*, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesBundle:
    trend_direction = TrendDirectionSeriesClassifier().build(context, feature_store)
    trend_character = TrendCharacterSeriesClassifier().build(context, feature_store)
    volatility_state = VolatilitySeriesClassifier().build(context, feature_store)
    breadth_state = BreadthSeriesClassifier().build(context, feature_store)
    event_calendar = build_event_calendar_series(context)
    network_fragility = NetworkFragilitySeriesClassifier().build(
        context,
        feature_store,
        breadth_active_labels_by_date=breadth_state.active_labels_by_date,
        volatility_active_labels_by_date=volatility_state.active_labels_by_date,
    )
    return AxisSeriesBundle(
        trend_direction=trend_direction,
        trend_character=trend_character,
        volatility_state=volatility_state,
        breadth_state=breadth_state,
        event_calendar=event_calendar,
        network_fragility=network_fragility,
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
