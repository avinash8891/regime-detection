from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from datetime import date
from typing import cast

import pandas as pd

from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.axis_builders.per_label import build_per_label_axis_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import DataQuality, VolumeLiquidityStateOutput
from regime_detection.volume_liquidity_rules import (
    VOLUME_LIQUIDITY_RISK_RANK,
    VolumeLiquidityLabel,
    VolumeLiquidityRuleInputs,
    evaluate_rules_with_evidence as evaluate_volume_liquidity_rules_with_evidence,
)

# V2 §1E volume/liquidity gate follows the existing 20d z-score cold start.
VOLUME_LIQUIDITY_REQUIRED_TRADING_DAYS = 20


def _volume_liquidity_output(
    *,
    raw_label: str,
    stable_label: str,
    active_label: str,
    evidence: dict[str, object],
    data_quality: DataQuality,
) -> VolumeLiquidityStateOutput:
    return VolumeLiquidityStateOutput(
        raw_label=cast(VolumeLiquidityLabel, raw_label),
        stable_label=cast(VolumeLiquidityLabel, stable_label),
        active_label=cast(VolumeLiquidityLabel, active_label),
        evidence=evidence,
        data_quality=data_quality,
    )


def build_volume_liquidity_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, VolumeLiquidityStateOutput] | None:
    """V2 §1E volume/liquidity axis classifier."""
    volume_features = feature_store.volume_liquidity_v2
    if volume_features is None:
        return None

    volume_liquidity_config = context.config.volume_liquidity_state
    if volume_liquidity_config is None:
        # Defensive: feature seam present but classifier config missing.
        return None

    # `return_1d` is the V1 single source of truth (documented implementation decision).
    return_1d_series = feature_store.volatility.return_1d
    volume_zscore_series = volume_features.volume_zscore_20d

    # documented implementation decision: optional §1C volatility_state_v2 percentiles feed
    # `liquidity_gap_behavior` when available. When absent, rules see NaN
    # and fall through per V1 §2.7 cold-start semantics.
    volatility_v2 = feature_store.volatility_state_v2
    gap_freq_pct_series: pd.Series | None = None
    gap_freq_series: pd.Series | None = None
    intraday_range_series: pd.Series | None = None
    intraday_pct_series: pd.Series | None = None
    if volatility_v2 is not None:
        gap_freq_series = volatility_v2.gap_frequency_20d
        gap_freq_pct_series = volatility_v2.gap_frequency_percentile_252d
        intraday_range_series = volatility_v2.intraday_range
        intraday_pct_series = volatility_v2.intraday_range_percentile_252d

    required_inputs: list[pd.Series] = [
        volume_zscore_series,
        return_1d_series,
    ]
    if gap_freq_pct_series is not None and intraday_pct_series is not None:
        required_inputs.extend([gap_freq_pct_series, intraday_pct_series])
    required_trading_days = VOLUME_LIQUIDITY_REQUIRED_TRADING_DAYS
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
            raw_label=None,
            max_freshness_days=max_freshness_days,
            min_completeness=min_completeness,
        )

        if quality_forces_unknown(day_quality):
            raw_labels.append("unknown")
            per_day_data_quality.append(day_quality)
            per_day_evidence.append(
                {"reason": day_quality.reason or "insufficient_data"}
            )
            continue

        volume_zscore_20d = (
            float(volume_zscore_series.loc[dt])
            if dt in volume_zscore_series.index
            else float("nan")
        )
        return_1d = (
            float(return_1d_series.loc[dt])
            if dt in return_1d_series.index
            else float("nan")
        )
        gap_freq_pct = (
            float(gap_freq_pct_series.loc[dt])
            if gap_freq_pct_series is not None and dt in gap_freq_pct_series.index
            else float("nan")
        )
        gap_freq_20d = (
            float(gap_freq_series.loc[dt])
            if gap_freq_series is not None and dt in gap_freq_series.index
            else float("nan")
        )
        intraday_range = (
            float(intraday_range_series.loc[dt])
            if intraday_range_series is not None and dt in intraday_range_series.index
            else float("nan")
        )
        intraday_pct = (
            float(intraday_pct_series.loc[dt])
            if intraday_pct_series is not None and dt in intraday_pct_series.index
            else float("nan")
        )

        inputs = VolumeLiquidityRuleInputs(
            volume_zscore_20d=volume_zscore_20d,
            return_1d=return_1d,
            gap_frequency_percentile_252d=gap_freq_pct,
            intraday_range_percentile_252d=intraday_pct,
            gap_frequency_20d=gap_freq_20d,
            intraday_range=intraday_range,
        )
        rule_evaluation = evaluate_volume_liquidity_rules_with_evidence(
            inputs=inputs,
            config=volume_liquidity_config.rules,
        )
        label = rule_evaluation.label
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append(
            {
                "rule_evidence": {
                    "volume_zscore_20d": float(f"{volume_zscore_20d:.8g}"),
                    "return_1d": float(f"{return_1d:.8g}"),
                    "gap_frequency_20d": float(f"{gap_freq_20d:.8g}"),
                    "gap_frequency_percentile_252d": float(f"{gap_freq_pct:.8g}"),
                    "intraday_range": float(f"{intraday_range:.8g}"),
                    "intraday_range_percentile_252d": float(f"{intraday_pct:.8g}"),
                },
                "rule_path": rule_evaluation.rule_path,
                "rule_reason": rule_evaluation.reason,
            }
        )

    return build_per_label_axis_outputs(
        sessions=context.sessions,
        raw_labels=raw_labels,
        risk_rank=VOLUME_LIQUIDITY_RISK_RANK,
        deescalation_days_by_label=volume_liquidity_config.deescalation_days_by_label,
        default_deescalation_days=volume_liquidity_config.default_deescalation_days,
        max_unknown_freeze_days=volume_liquidity_config.max_unknown_freeze_days,
        data_quality=per_day_data_quality,
        evidence=per_day_evidence,
        output_factory=_volume_liquidity_output,
    )
