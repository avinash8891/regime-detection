from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from regime_detection.trend_character import (
    _RISK_RANK as TREND_CHARACTER_RISK_RANK,
    build_raw_outputs as build_trend_character_raw_outputs,
)

if TYPE_CHECKING:
    from regime_detection.axis_series import AxisSeriesResult
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


# 63-session warm-up matches the longest required window — V1
# prior_63d_drawdown and V2 range_bound return_63d.
TREND_CHARACTER_REQUIRED_TRADING_DAYS = 63


def build_trend_character_axis_series(
    context: MarketContext, feature_store: FeatureStore
) -> AxisSeriesResult:
    close = context.spy_ohlcv["close"]
    close_index = pd.DatetimeIndex(close.index)
    features = feature_store.trend_character
    tc_v2_config = context.config.trend_character_v2
    if tc_v2_config is None:
        raise RuntimeError("trend_character_v2 is required")
    hysteresis_config = context.config.trend_character
    raw_labels, raw_evidence = build_trend_character_raw_outputs(
        features,
        allow_v2_labels=context.config.config_version != "core3-v1.0.0",
        followthrough_rate_threshold=tc_v2_config.followthrough_rate_threshold,
        range_bound_return_63d_threshold=tc_v2_config.range_bound_return_63d_threshold,
        range_bound_midpoint_excursion_threshold=tc_v2_config.range_bound_midpoint_excursion_threshold,
        range_bound_adx_threshold=tc_v2_config.range_bound_adx_threshold,
    )
    from regime_detection.axis_series import _build_axis_outputs

    return _build_axis_outputs(
        dates=[ts.date() for ts in close_index],
        raw_labels=raw_labels,
        raw_evidence=raw_evidence,
        risk_rank=TREND_CHARACTER_RISK_RANK,
        deescalation_days_by_label=hysteresis_config.deescalation_days_by_label,
        escalation_days_by_label=hysteresis_config.escalation_days_by_label,
        default_escalation_days=hysteresis_config.default_escalation_days,
        default_deescalation_days=hysteresis_config.default_deescalation_days,
        max_unknown_freeze_days=hysteresis_config.max_unknown_freeze_days,
        required_inputs=[close, context.spy_ohlcv["high"], context.spy_ohlcv["low"]],
        required_trading_days=TREND_CHARACTER_REQUIRED_TRADING_DAYS,
        max_freshness_days=context.config.data_quality.max_freshness_days,
        min_completeness=context.config.data_quality.min_completeness,
    )
