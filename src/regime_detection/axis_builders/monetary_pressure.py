from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd

from regime_detection.central_bank_text import CENTRAL_BANK_TEXT_EVIDENCE_QUALITY
from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.axis_builders.per_label import build_per_label_axis_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.market_context import MarketContext
from regime_detection.models import DataQuality, MonetaryPressureV2Output
from regime_detection.monetary_pressure import (
    MONETARY_PRESSURE_V2_RISK_RANK,
    MonetaryPressureV2Label,
    build_rule_inputs_for_date as build_monetary_pressure_rule_inputs_for_date,
    evaluate_rules as evaluate_monetary_pressure_rules,
)


def _monetary_pressure_required_trading_days(
    config: object,
) -> int:
    """Trailing feature-output sessions required by the §2A quality gate.

    The z-score normalizer warmup is already represented by leading NaNs in
    the computed feature series. This window checks that the post-warmup
    feature outputs are continuously available over their longest lookback.
    """
    return int(
        max(
            getattr(config, "yield_change_lookback_days"),
            getattr(config, "rate_shock_lookback_days"),
            getattr(config, "broad_usd_lookback_days"),
        )
    )


def _monetary_pressure_output(
    *,
    raw_label: str,
    stable_label: str,
    active_label: str,
    evidence: dict[str, object],
    data_quality: DataQuality,
) -> MonetaryPressureV2Output:
    return MonetaryPressureV2Output(
        raw_label=cast(MonetaryPressureV2Label, raw_label),
        stable_label=cast(MonetaryPressureV2Label, stable_label),
        active_label=cast(MonetaryPressureV2Label, active_label),
        evidence=evidence,
        data_quality=data_quality,
    )


def build_monetary_pressure_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, MonetaryPressureV2Output] | None:
    """V2 §2A monetary pressure axis classifier (implementation decision)."""
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
    required_trading_days = _monetary_pressure_required_trading_days(
        v2_features_config
    )
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
            cb_val = features.central_bank_text_score.get(dt)
            score = float(cb_val) if pd.notna(cb_val) else None
            per_day_evidence[-1]["rule_evidence"]["central_bank_text_score"] = (
                score
            )
            per_day_evidence[-1]["central_bank_text_evidence"] = {
                "score": score,
                **CENTRAL_BANK_TEXT_EVIDENCE_QUALITY,
            }

    return build_per_label_axis_outputs(
        sessions=context.sessions,
        raw_labels=raw_labels,
        risk_rank=MONETARY_PRESSURE_V2_RISK_RANK,
        deescalation_days_by_label=mp_config.deescalation_days_by_label,
        default_deescalation_days=mp_config.default_deescalation_days,
        max_unknown_freeze_days=mp_config.max_unknown_freeze_days,
        data_quality=per_day_data_quality,
        evidence=per_day_evidence,
        output_factory=_monetary_pressure_output,
    )
