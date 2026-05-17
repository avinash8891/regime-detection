from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd

from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.feature_store import FeatureStore
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.market_context import MarketContext
from regime_detection.models import DataQuality, MonetaryPressureV2Output
from regime_detection.monetary_pressure import (
    MONETARY_PRESSURE_V2_RISK_RANK,
    MonetaryPressureV2Label,
    build_rule_inputs_for_date as build_monetary_pressure_rule_inputs_for_date,
    evaluate_rules as evaluate_monetary_pressure_rules,
)


# V2 §2A feature series already encode their longer warm-ups as NaN.
MONETARY_PRESSURE_REQUIRED_TRADING_DAYS = 1


def build_monetary_pressure_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, MonetaryPressureV2Output] | None:
    """V2 §2A monetary pressure axis classifier (Ambiguity Log #46)."""
    features = feature_store.monetary
    if features is None:
        return None
    mp_config = context.config.monetary_pressure_state
    if mp_config is None:
        return None

    v2_features_config = context.config.monetary_pressure_v2
    if v2_features_config is None:
        return None

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
        dt = cast(pd.Timestamp, pd.Timestamp(day))

        day_quality = assess_series_input_quality(
            as_of_date=day,
            required_inputs=required_inputs,
            required_trading_days=MONETARY_PRESSURE_REQUIRED_TRADING_DAYS,
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

        inputs = build_monetary_pressure_rule_inputs_for_date(features=features, dt=dt)
        label = evaluate_monetary_pressure_rules(inputs=inputs, config=mp_config.rules)
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
        if features.central_bank_text_score is not None:
            per_day_evidence[-1]["rule_evidence"]["central_bank_text_score"] = (
                float(features.central_bank_text_score.loc[dt])
                if pd.notna(features.central_bank_text_score.loc[dt])
                else None
            )

    stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
        raw_labels=list(raw_labels),
        risk_rank=cast(dict[str, int], MONETARY_PRESSURE_V2_RISK_RANK),
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
            raw_label=cast(MonetaryPressureV2Label, raw),
            stable_label=cast(MonetaryPressureV2Label, stable),
            active_label=cast(MonetaryPressureV2Label, active),
            evidence=evidence,
            data_quality=dq,
        )
    return outputs
