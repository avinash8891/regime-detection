from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from regime_detection.trend_direction import (
    _RISK_RANK as TREND_DIRECTION_RISK_RANK,
    apply_hysteresis as apply_trend_direction_hysteresis,
    build_raw_outputs as build_trend_direction_raw_outputs,
)

if TYPE_CHECKING:
    from regime_detection.axis_series import AxisSeriesResult
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


# V1 trend-direction warm-up follows the long-MA calibration used by the
# existing data-quality gate.
TREND_DIRECTION_REQUIRED_TRADING_DAYS = 200


def build_trend_direction_axis_series(
    context: MarketContext, feature_store: FeatureStore
) -> AxisSeriesResult:
    close = context.spy_ohlcv["close"]
    close_index = pd.DatetimeIndex(close.index)
    features = feature_store.trend_direction
    # implementation phase — thread v2 §1A features + rules through when the v2 seam
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
        dates=close_index,
        raw_labels=raw_labels,
        escalation_days=context.config.hysteresis.trend_direction_escalation_days,
        deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
    )
    from regime_detection.axis_series import _build_axis_outputs

    return _build_axis_outputs(
        dates=[ts.date() for ts in close_index],
        raw_labels=raw_labels,
        stable_labels=stable_labels,
        active_labels=active_labels,
        raw_evidence=raw_evidence,
        risk_rank=TREND_DIRECTION_RISK_RANK,
        deescalation_days=context.config.hysteresis.trend_direction_deescalation_days,
        required_inputs=[close],
        required_trading_days=TREND_DIRECTION_REQUIRED_TRADING_DAYS,
        max_freshness_days=context.config.data_quality.max_freshness_days,
        min_completeness=context.config.data_quality.min_completeness,
    )
