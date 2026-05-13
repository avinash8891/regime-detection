from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from regime_detection.breadth_state import (
    _RISK_RANK as BREADTH_RISK_RANK,
    _data_quality_for_asof as breadth_data_quality_for_asof,
    _evaluate_broadening_breadth,
    _evaluate_narrowing_breadth,
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
from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CreditFundingLabel,
    RULE_PRECEDENCE as CREDIT_FUNDING_RULE_PRECEDENCE,
    build_rule_inputs_for_date as build_credit_funding_rule_inputs_for_date,
    evaluate_rules as evaluate_credit_funding_rules,
)
from regime_detection.models import (
    AxisOutput,
    BreadthStateOutput,
    CreditFundingOutput,
    DataQuality,
    EventCalendarOutput,
    MonetaryPressureV2Output,
    NetworkFragilityOutput,
    VolumeLiquidityStateOutput,
)
from regime_detection.monetary_pressure import (
    MONETARY_PRESSURE_V2_RISK_RANK,
    MonetaryPressureV2Label,
    build_rule_inputs_for_date as build_monetary_pressure_rule_inputs_for_date,
    evaluate_rules as evaluate_monetary_pressure_rules,
)
from regime_detection.network_fragility_rules import (
    NETWORK_FRAGILITY_RISK_RANK,
    NetworkFragilityLabel,
    build_rule_inputs_for_date,
    evaluate_rules,
)
from regime_detection.volume_liquidity_rules import (
    VOLUME_LIQUIDITY_RISK_RANK,
    VolumeLiquidityLabel,
    VolumeLiquidityRuleInputs,
    evaluate_rules as evaluate_volume_liquidity_rules,
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
    # V2 §1E volume/liquidity — None in pure-v1 mode (no v2 config),
    # populated by VolumeLiquidityStateSeriesClassifier when feature_store
    # has the v2 volume_liquidity_v2 seam (Slice 2.7).
    volume_liquidity_state: dict[date, VolumeLiquidityStateOutput] | None = None
    # V2 §2C credit/funding — None in pure-v1 mode (no v2 config),
    # populated by CreditFundingSeriesClassifier when feature_store has
    # the credit_funding seam lit (Slice 4).
    credit_funding: dict[date, CreditFundingOutput] | None = None
    # V2 §2A monetary pressure — None in pure-v1 mode (no v2 config), populated
    # by MonetaryPressureV2SeriesClassifier when feature_store.monetary is lit
    # AND context.config.monetary_pressure_state is non-None (Ambiguity Log #46).
    monetary_pressure_state: dict[date, MonetaryPressureV2Output] | None = None


class AxisSeriesClassifier(Protocol):
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult: ...


class TrendDirectionSeriesClassifier:
    def build(self, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesResult:
        close = context.spy_ohlcv["close"]
        features = feature_store.trend_direction
        # Slice 2.5 — thread v2 §1A features + rules through when the v2 seam
        # is populated. When the v2 config is absent (v1-only callers), the
        # arguments stay None and v1 byte-identity is preserved by
        # build_raw_outputs (see test_trend_direction_v2_recovery_rule).
        trend_v2_features = feature_store.trend_direction_v2
        trend_v2_config = context.config.trend_direction_v2
        trend_v2_rules = trend_v2_config.rules if trend_v2_config is not None else None
        raw_labels, raw_evidence = build_trend_direction_raw_outputs(
            features,
            trend_direction_v2_features=trend_v2_features,
            trend_direction_v2_rules=trend_v2_rules,
        )
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
        # Slice 2.6 — thread v2 §1C features + rules through when the v2
        # seam is populated. When the v2 config is absent (v1-only callers),
        # the arguments stay None and v1 byte-identity is preserved by
        # build_raw_outputs (see test_volatility_state_v2_rising_vol_rule).
        vol_v2_features = feature_store.volatility_state_v2
        vol_v2_config = context.config.volatility_state_v2
        vol_v2_rules = vol_v2_config.rules if vol_v2_config is not None else None
        raw_labels, raw_evidence = build_volatility_raw_outputs(
            features,
            volatility_state_v2_features=vol_v2_features,
            volatility_state_v2_rules=vol_v2_rules,
        )
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

        # V2 §1D extension (Ambiguity Log #21-#26, #68): when the PIT seam is
        # lit AND ALL four required PIT features are non-None, evaluate the
        # narrowing_breadth and broadening_breadth predicates per session and
        # apply the spec §1D line 284 precedence walk. When the PIT seam is
        # unlit (default-config callers, no PIT inputs), V2 rules silently do
        # NOT fire — V1 byte-identity is preserved (see Hard Constraint #1).
        v2_features = feature_store.breadth_state_v2
        v2_config = context.config.breadth_state_v2
        v2_active = (
            v2_features is not None
            and v2_config is not None
            and v2_features.pct_above_50dma is not None
            and v2_features.pct_above_200dma is not None
            and v2_features.nh_nl_ratio is not None
            and v2_features.ad_line_slope_20d is not None
        )
        if v2_active:
            assert v2_features is not None  # narrowing for type-checker
            assert v2_config is not None
            lookback = v2_config.label_rate_of_change_lookback_sessions
            nh_nl_threshold = v2_config.nh_nl_ratio_narrowing_threshold
            updated_labels: list[str] = []
            for idx_pos, day in enumerate(spy_close.index):
                v1_raw = raw_labels[idx_pos]
                narrowing_fires = _evaluate_narrowing_breadth(
                    pct_above_50dma=v2_features.pct_above_50dma,
                    pct_above_200dma=v2_features.pct_above_200dma,
                    nh_nl_ratio=v2_features.nh_nl_ratio,
                    dt=day,
                    lookback_sessions=lookback,
                    nh_nl_threshold=nh_nl_threshold,
                )
                broadening_fires = _evaluate_broadening_breadth(
                    nh_nl_ratio=v2_features.nh_nl_ratio,
                    ad_line_slope_20d=v2_features.ad_line_slope_20d,
                    dt=day,
                    lookback_sessions=lookback,
                )
                # Precedence walker (spec §1D line 284). breadth_thrust and
                # recovery_breadth slots are reserved but never fire today
                # (Ambiguity Log #69 / #70 — labels DEFERRED).
                if v1_raw == "divergent_fragile":
                    resolved = "divergent_fragile"
                elif narrowing_fires:
                    resolved = "narrowing_breadth"
                elif v1_raw in {"weak_breadth", "healthy_breadth", "neutral_breadth", "unknown"} and broadening_fires:
                    resolved = "broadening_breadth"
                else:
                    resolved = v1_raw
                if resolved != v1_raw:
                    raw_evidence[idx_pos] = {
                        **raw_evidence[idx_pos],
                        "v2_narrowing_breadth": narrowing_fires,
                        "v2_broadening_breadth": broadening_fires,
                        "v1_raw_label": v1_raw,
                    }
                updated_labels.append(resolved)
            raw_labels = updated_labels  # type: ignore[assignment]

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
            # Pure-quality assessment: we compute the raw label AFTER quality
            # (rule engine, below) and re-check with quality_forces_unknown.
            # `skip_raw_label_short_circuit=True` keeps the helper from
            # collapsing to "insufficient_history" before we have a label.
            day_quality = assess_series_input_quality(
                as_of_date=day,
                required_inputs=required_inputs,
                required_trading_days=required_trading_days,
                raw_label="",
                max_freshness_days=max_freshness_days,
                min_completeness=min_completeness,
                skip_raw_label_short_circuit=True,
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

            # I1: strict V1 axis alignment. When the caller supplied a v1
            # dict, missing-day → KeyError (loud) rather than silent
            # "unknown" substitution which would defang systemic_stress on
            # any drifted session. The "unknown" fallback applies ONLY when
            # the caller explicitly omitted the v1 dict (unit-test path).
            if breadth_active_labels_by_date is None:
                breadth_label = "unknown"
            else:
                if day not in breadth_active_labels_by_date:
                    raise KeyError(
                        f"breadth_active_labels_by_date missing session {day!r} "
                        "(v1/v2 calendar drift would silently downgrade rules to 'unknown')"
                    )
                breadth_label = breadth_active_labels_by_date[day]
            if volatility_active_labels_by_date is None:
                volatility_label = "unknown"
            else:
                if day not in volatility_active_labels_by_date:
                    raise KeyError(
                        f"volatility_active_labels_by_date missing session {day!r} "
                        "(v1/v2 calendar drift would silently downgrade rules to 'unknown')"
                    )
                volatility_label = volatility_active_labels_by_date[day]

            label = evaluate_rules(
                inputs=rule_inputs,
                config=network_fragility_config.rules,
                breadth_label=breadth_label,  # type: ignore[arg-type]
                volatility_label=volatility_label,  # type: ignore[arg-type]
                # TODO(slice-4): wire credit_funding.active_label per v2 §2C.
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
            default_deescalation_days=network_fragility_config.default_deescalation_days,
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


class VolumeLiquidityStateSeriesClassifier:
    """V2 §1E volume/liquidity axis classifier (Slice 2.7).

    Pipeline:

      1. Read pre-computed ``volume_zscore_20d`` from
         ``feature_store.volume_liquidity_v2`` (slice 2.4). If the seam
         is None (no v2 config / no volume column) return None — the
         timeline then leaves ``RegimeOutput.volume_liquidity_state``
         as ``None`` and the V1 wire contract is preserved.
      2. Pull ``return_1d`` from the V1 ``feature_store.volatility``
         (single source of truth — see Ambiguity Log #42).
      3. Per session, assess data quality (``assess_series_input_quality``
         + ``quality_forces_unknown``). Quality failures force ``unknown``.
      4. Evaluate ``volume_liquidity_rules.evaluate_rules`` to produce
         the raw label per §1E precedence
         (``panic_volume > liquidity_gap_behavior(deferred) > normal_volume > unknown``).
      5. Apply per-label asymmetric hysteresis (Ambiguity Log #41).
      6. Emit one ``VolumeLiquidityStateOutput`` per session.
    """

    def build(
        self,
        context: MarketContext,
        feature_store: FeatureStore,
    ) -> dict[date, VolumeLiquidityStateOutput] | None:
        volume_features = feature_store.volume_liquidity_v2
        if volume_features is None:
            return None

        volume_liquidity_config = context.config.volume_liquidity_state
        if volume_liquidity_config is None:
            # Defensive: feature seam present but classifier config missing.
            return None

        # `return_1d` is the V1 single source of truth (Ambiguity Log #42).
        # The slice-2.4 volume_zscore_20d feature shares the SPY index with
        # the V1 volatility features, so direct .loc access on the same dt
        # is safe — no reindex needed.
        return_1d_series = feature_store.volatility.return_1d
        volume_zscore_series = volume_features.volume_zscore_20d

        required_inputs: list[pd.Series] = [
            volume_zscore_series,
            return_1d_series,
        ]
        # The 20d z-score is the binding cold-start window. Once it has
        # 20 sessions of data the rules can fire; the engine's outer
        # ENGINE_MINIMUM_HISTORY (320) already comfortably exceeds this.
        required_trading_days = 20
        max_freshness_days = context.config.data_quality.max_freshness_days
        min_completeness = context.config.data_quality.min_completeness

        raw_labels: list[VolumeLiquidityLabel] = []
        per_day_data_quality: list[DataQuality] = []
        per_day_evidence: list[dict[str, object]] = []

        for day in context.sessions:
            dt = pd.Timestamp(day)

            day_quality = assess_series_input_quality(
                as_of_date=day,
                required_inputs=required_inputs,
                required_trading_days=required_trading_days,
                raw_label="",
                max_freshness_days=max_freshness_days,
                min_completeness=min_completeness,
                skip_raw_label_short_circuit=True,
            )

            if quality_forces_unknown(day_quality):
                raw_labels.append("unknown")
                per_day_data_quality.append(day_quality)
                per_day_evidence.append({"reason": day_quality.reason or "insufficient_data"})
                continue

            # NaN-safe scalar materialization. The deferred liquidity_gap
            # percentile inputs are not yet available — pass NaN so the
            # rule signature stays forward-compat without depending on
            # data the feature store does not yet expose.
            volume_zscore_20d = float(volume_zscore_series.loc[dt]) if dt in volume_zscore_series.index else float("nan")
            return_1d = float(return_1d_series.loc[dt]) if dt in return_1d_series.index else float("nan")

            inputs = VolumeLiquidityRuleInputs(
                volume_zscore_20d=volume_zscore_20d,
                return_1d=return_1d,
                gap_frequency_percentile_252d=float("nan"),  # deferred (Ambiguity Log #40)
                intraday_range_percentile_252d=float("nan"),  # deferred
            )
            label = evaluate_volume_liquidity_rules(
                inputs=inputs,
                config=volume_liquidity_config.rules,
            )
            raw_labels.append(label)
            per_day_data_quality.append(day_quality)
            # Round evidence floats to 8 significant digits to absorb
            # pandas-rolling accumulation drift that depends on the size
            # of the input window slice (the same as-of-day value can
            # differ at ~1e-11 between callers that pre-slice the context
            # at different lookbacks). 8 sig-figs is well above any
            # threshold the rules care about (2.0 / -0.02) but trims the
            # noise so the wire is reproducible across call paths.
            per_day_evidence.append({
                "rule_evidence": {
                    "volume_zscore_20d": float(f"{volume_zscore_20d:.8g}"),
                    "return_1d": float(f"{return_1d:.8g}"),
                },
                "deferred_inputs": {
                    "gap_frequency_percentile_252d": "deferred (Ambiguity Log #40)",
                    "intraday_range_percentile_252d": "deferred (Ambiguity Log #40)",
                },
            })

        stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=VOLUME_LIQUIDITY_RISK_RANK,
            deescalation_days_by_label=volume_liquidity_config.deescalation_days_by_label,
            default_deescalation_days=volume_liquidity_config.default_deescalation_days,
        )

        outputs: dict[date, VolumeLiquidityStateOutput] = {}
        for day, raw, stable, active, dq, evidence in zip(
            context.sessions,
            raw_labels,
            stable_labels,
            active_labels,
            per_day_data_quality,
            per_day_evidence,
            strict=True,
        ):
            outputs[day] = VolumeLiquidityStateOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence=evidence,
                data_quality=dq,
            )
        return outputs


class CreditFundingSeriesClassifier:
    """V2 §2C credit/funding axis classifier (Slice 4).

    Pipeline:

      1. Read pre-computed features from ``feature_store.credit_funding``
         (compute_credit_funding_features). If the seam is None (no v2
         config / required inputs absent) return None — the timeline then
         leaves ``RegimeOutput.credit_funding_state`` as ``None``.
      2. Per session, run the §2C unknown gate (spec lines 2122-2126):
         - HYG/LQD/TLT stale > 5 sessions → unknown
         - NFCI stale > 14 days → unknown
         - SOFR or IORB missing on session → unknown
         - assess_series_input_quality fails on any required series → unknown
      3. Materialize per-day scalar rule inputs (build_rule_inputs_for_date),
         then evaluate §2C precedence (deleveraging > funding_squeeze >
         credit_stress > spread_widening > credit_calm > unknown).
      4. Apply per-label asymmetric hysteresis (§2C lines 2105-2118).
      5. Emit one CreditFundingOutput per session.
    """

    def build(
        self,
        context: MarketContext,
        feature_store: FeatureStore,
    ) -> dict[date, CreditFundingOutput] | None:
        features = feature_store.credit_funding
        if features is None:
            return None
        cf_config = context.config.credit_funding
        if cf_config is None:
            return None

        spy_close = context.spy_ohlcv["close"]
        volatility_features = feature_store.volatility
        realized_vol_pct = volatility_features.realized_vol_percentile_252d
        nf_features = feature_store.network_fragility
        if nf_features is None:
            avg_corr_pct_series = pd.Series(float("nan"), index=spy_close.index)
        else:
            avg_corr_pct_series = nf_features.avg_pairwise_corr_percentile_504d

        # The credit_funding seam guarantees these series exist on the
        # SPY index when feature_store.credit_funding is non-None.
        cross_asset_closes = context.cross_asset_closes or {}
        macro_series = context.macro_series or {}
        hyg_close = cross_asset_closes.get("HYG")
        lqd_close = cross_asset_closes.get("LQD")
        tlt_close = cross_asset_closes.get("TLT")
        sofr_series = macro_series.get("SOFR")
        iorb_series = macro_series.get("IORB")
        nfci_series = macro_series.get("NFCI")

        # Quality-gate primary inputs. Lookback gates on the 504d percentile
        # window — the longest binding cold-start for any rule predicate.
        required_inputs: list[pd.Series] = [
            features.hy_spread_proxy_63d,
            features.ig_spread_proxy_63d,
            features.kre_spy_ratio,
            features.sofr_iorb_spread,
            spy_close,
        ]
        required_trading_days = cf_config.rules.hy_percentile_504d_lookback
        max_freshness_days = context.config.data_quality.max_freshness_days
        min_completeness = context.config.data_quality.min_completeness

        raw_labels: list[CreditFundingLabel] = []
        per_day_data_quality: list[DataQuality] = []
        per_day_evidence: list[dict[str, object]] = []

        nfci_carried = features.nfci_daily_carried

        for day in context.sessions:
            dt = pd.Timestamp(day)

            # §2C unknown gate (spec lines 2122-2126). Run BEFORE the generic
            # assess_series_input_quality so the §2C-specific staleness
            # messages reach evidence.
            etf_staleness_breach = False
            etf_stale_label: str | None = None
            for etf_label, etf_series in (
                ("HYG", hyg_close),
                ("LQD", lqd_close),
                ("TLT", tlt_close),
            ):
                if etf_series is None:
                    etf_staleness_breach = True
                    etf_stale_label = etf_label
                    break
                staleness = _trailing_staleness_sessions(etf_series, dt)
                if staleness > cf_config.etf_stale_sessions:
                    etf_staleness_breach = True
                    etf_stale_label = etf_label
                    break

            sofr_missing = (
                sofr_series is None
                or dt not in sofr_series.index
                or pd.isna(sofr_series.loc[dt])
            )
            iorb_missing = (
                iorb_series is None
                or dt not in iorb_series.index
                or pd.isna(iorb_series.loc[dt])
            )
            nfci_staleness_days = _nfci_calendar_staleness_days(nfci_series, dt)
            nfci_stale = nfci_staleness_days > cf_config.nfci_stale_days

            if etf_staleness_breach or sofr_missing or iorb_missing or nfci_stale:
                reason_parts: list[str] = []
                if etf_staleness_breach:
                    reason_parts.append(f"etf_stale:{etf_stale_label}")
                if sofr_missing:
                    reason_parts.append("sofr_missing")
                if iorb_missing:
                    reason_parts.append("iorb_missing")
                if nfci_stale:
                    reason_parts.append(f"nfci_stale_{nfci_staleness_days}d")
                gate_reason = ",".join(reason_parts)
                raw_labels.append("unknown")
                per_day_data_quality.append(
                    DataQuality(
                        status="stale_data" if (etf_staleness_breach or nfci_stale) else "insufficient_data",
                        freshness_days=None,
                        completeness=None,
                        reason=gate_reason,
                    )
                )
                per_day_evidence.append({"reason": gate_reason})
                continue

            day_quality = assess_series_input_quality(
                as_of_date=day,
                required_inputs=required_inputs,
                required_trading_days=required_trading_days,
                raw_label="",
                max_freshness_days=max_freshness_days,
                min_completeness=min_completeness,
                skip_raw_label_short_circuit=True,
            )
            if quality_forces_unknown(day_quality):
                raw_labels.append("unknown")
                per_day_data_quality.append(day_quality)
                per_day_evidence.append({"reason": day_quality.reason or "insufficient_data"})
                continue

            rule_inputs = build_credit_funding_rule_inputs_for_date(
                features=features,
                dt=dt,
                realized_vol_21d_percentile_252d=realized_vol_pct,
                avg_pairwise_corr_percentile_504d=avg_corr_pct_series,
            )
            label = evaluate_credit_funding_rules(
                inputs=rule_inputs,
                config=cf_config.rules,
            )
            raw_labels.append(label)
            per_day_data_quality.append(day_quality)
            per_day_evidence.append(
                {
                    "rule_evidence": {
                        "hy_spread_proxy_percentile_504d": rule_inputs.hy_spread_proxy_percentile_504d,
                        "hy_spread_proxy_slope_21d": rule_inputs.hy_spread_proxy_slope_21d,
                        "ig_spread_proxy_slope_21d": rule_inputs.ig_spread_proxy_slope_21d,
                        "broad_usd_index_zscore_21d": rule_inputs.broad_usd_index_zscore_21d,
                        "sofr_iorb_slope_21d": rule_inputs.sofr_iorb_slope_21d,
                        "spy_21d_return": rule_inputs.spy_21d_return,
                        "tlt_21d_return": rule_inputs.tlt_21d_return,
                        "realized_vol_21d_percentile_252d": rule_inputs.realized_vol_21d_percentile_252d,
                        "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
                    },
                    "nfci_daily_carried": _safe_float(nfci_carried, dt),
                    "kre_spy_slope_63d": _safe_float(features.kre_spy_slope_63d, dt),
                    "bias_warning_code": "credit_spread_proxy_total_return_differential",
                }
            )

        stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=CREDIT_FUNDING_RISK_RANK,
            deescalation_days_by_label=cf_config.deescalation_days_by_label,
            default_deescalation_days=cf_config.default_deescalation_days,
        )

        outputs: dict[date, CreditFundingOutput] = {}
        for day, raw, stable, active, dq, evidence in zip(
            context.sessions,
            raw_labels,
            stable_labels,
            active_labels,
            per_day_data_quality,
            per_day_evidence,
            strict=True,
        ):
            outputs[day] = CreditFundingOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence=evidence,
                data_quality=dq,
            )
        return outputs


class MonetaryPressureV2SeriesClassifier:
    """V2 §2A monetary pressure axis classifier (Ambiguity Log #46).

    Pipeline mirrors VolumeLiquidityStateSeriesClassifier:

      1. Read pre-computed features from ``feature_store.monetary``. If the
         seam is None (no v2 config / no DGS2+DGS10) return None so the
         timeline leaves ``RegimeOutput.monetary_pressure_state`` as None.
      2. Per session, assess data quality (``assess_series_input_quality`` +
         ``quality_forces_unknown``). Quality failures force ``unknown``.
      3. Materialize per-day scalar inputs and evaluate the §2A rule
         precedence (rate_shock > tightening_pressure > easing_pressure >
         neutral_monetary).
      4. Apply per-label asymmetric hysteresis per Log #46 (e).
      5. Emit one ``MonetaryPressureV2Output`` per session.
    """

    def build(
        self,
        context: MarketContext,
        feature_store: FeatureStore,
    ) -> dict[date, MonetaryPressureV2Output] | None:
        features = feature_store.monetary
        if features is None:
            return None
        mp_config = context.config.monetary_pressure_state
        if mp_config is None:
            return None

        # The 63d-change + 1260d-normalizer window is the binding cold-start
        # for the §2A rules. The 21d-change variant warms up sooner so the
        # 63d-derived series gate the data-quality assessment.
        v2_features_config = context.config.monetary_pressure_v2
        if v2_features_config is None:
            # Defensive: features seam lit but feature config missing.
            return None
        required_trading_days = (
            v2_features_config.yield_change_lookback_days
            + v2_features_config.zscore_normalizer_window_days
        )

        required_inputs: list[pd.Series] = [
            features.yield_change_zscore_2y_63d,
            features.yield_change_zscore_10y_63d,
            features.yield_change_zscore_21d_2y,
            features.yield_change_zscore_21d_10y,
        ]
        max_freshness_days = context.config.data_quality.max_freshness_days
        min_completeness = context.config.data_quality.min_completeness

        raw_labels: list[MonetaryPressureV2Label] = []
        per_day_data_quality: list[DataQuality] = []
        per_day_evidence: list[dict[str, object]] = []

        for day in context.sessions:
            dt = pd.Timestamp(day)

            day_quality = assess_series_input_quality(
                as_of_date=day,
                required_inputs=required_inputs,
                required_trading_days=required_trading_days,
                raw_label="",
                max_freshness_days=max_freshness_days,
                min_completeness=min_completeness,
                skip_raw_label_short_circuit=True,
            )
            if quality_forces_unknown(day_quality):
                raw_labels.append("unknown")
                per_day_data_quality.append(day_quality)
                per_day_evidence.append(
                    {"reason": day_quality.reason or "insufficient_data"}
                )
                continue

            inputs = build_monetary_pressure_rule_inputs_for_date(
                features=features, dt=dt
            )
            label = evaluate_monetary_pressure_rules(
                inputs=inputs, config=mp_config.rules
            )
            raw_labels.append(label)
            per_day_data_quality.append(day_quality)
            per_day_evidence.append(
                {
                    "rule_evidence": {
                        "yield_change_zscore_2y_63d": inputs.zscore_2y_63d,
                        "yield_change_zscore_10y_63d": inputs.zscore_10y_63d,
                        "broad_usd_index_zscore_63d": inputs.broad_usd_zscore_63d,
                        "yield_change_zscore_21d_2y": inputs.zscore_21d_2y,
                        "yield_change_zscore_21d_10y": inputs.zscore_21d_10y,
                    },
                }
            )

        stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=MONETARY_PRESSURE_V2_RISK_RANK,
            deescalation_days_by_label=mp_config.deescalation_days_by_label,
            default_deescalation_days=mp_config.default_deescalation_days,
        )

        outputs: dict[date, MonetaryPressureV2Output] = {}
        for day, raw, stable, active, dq, evidence in zip(
            context.sessions,
            raw_labels,
            stable_labels,
            active_labels,
            per_day_data_quality,
            per_day_evidence,
            strict=True,
        ):
            outputs[day] = MonetaryPressureV2Output(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence=evidence,
                data_quality=dq,
            )
        return outputs


def _trailing_staleness_sessions(series: pd.Series, dt: pd.Timestamp) -> int:
    """Trading-day distance from ``dt`` to the last non-NaN observation
    at or before ``dt`` in ``series``. Returns a huge value if no such
    observation exists (forces unknown gate trip)."""
    if dt not in series.index:
        # Find the last index value <= dt.
        sub = series.loc[:dt]
    else:
        sub = series.loc[:dt]
    last_valid = sub.last_valid_index()
    if last_valid is None:
        return 10**9
    # Trading-day distance: count of session-index positions between them.
    idx = series.index
    try:
        pos_now = idx.get_loc(dt)
        pos_last = idx.get_loc(last_valid)
    except KeyError:
        return 10**9
    return int(pos_now - pos_last)


def _nfci_calendar_staleness_days(
    nfci_series: pd.Series | None, dt: pd.Timestamp
) -> int:
    """Calendar-day distance from ``dt`` to the last non-NaN NFCI release.

    NFCI is weekly; staleness is measured in calendar days (spec line 2124).
    Returns a huge value if the series is None or has no non-NaN history.
    """
    if nfci_series is None:
        return 10**9
    sub = nfci_series.loc[:dt]
    last_valid = sub.last_valid_index()
    if last_valid is None:
        return 10**9
    return int((dt.normalize() - pd.Timestamp(last_valid).normalize()).days)


def _safe_float(series: pd.Series, dt: pd.Timestamp) -> float:
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)


def build_axis_series_bundle(*, context: MarketContext, feature_store: FeatureStore) -> AxisSeriesBundle:
    trend_direction = TrendDirectionSeriesClassifier().build(context, feature_store)
    trend_character = TrendCharacterSeriesClassifier().build(context, feature_store)
    volatility_state = VolatilitySeriesClassifier().build(context, feature_store)
    breadth_state = BreadthSeriesClassifier().build(context, feature_store)
    event_calendar = build_event_calendar_series(context)
    credit_funding = CreditFundingSeriesClassifier().build(context, feature_store)
    network_fragility = NetworkFragilitySeriesClassifier().build(
        context,
        feature_store,
        breadth_active_labels_by_date=breadth_state.active_labels_by_date,
        volatility_active_labels_by_date=volatility_state.active_labels_by_date,
    )
    volume_liquidity_state = VolumeLiquidityStateSeriesClassifier().build(
        context, feature_store
    )
    monetary_pressure_state = MonetaryPressureV2SeriesClassifier().build(
        context, feature_store
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
        monetary_pressure_state=monetary_pressure_state,
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
