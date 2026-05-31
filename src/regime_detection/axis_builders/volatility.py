from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from regime_detection.volatility_state import (
    _RISK_RANK as VOLATILITY_RISK_RANK,
    build_raw_outputs as build_volatility_raw_outputs,
)

if TYPE_CHECKING:
    from regime_detection.axis_series import AxisSeriesResult
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


# V1 volatility warm-up follows the existing 252-session percentile gate.
VOLATILITY_REQUIRED_TRADING_DAYS = 252


def build_volatility_axis_series(
    context: MarketContext, feature_store: FeatureStore
) -> AxisSeriesResult:
    close = context.spy_ohlcv["close"]
    close_index = pd.DatetimeIndex(close.index)
    features = feature_store.volatility
    # Thread v2 §1C features + rules through when the v2
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
    hysteresis_config = context.config.volatility_state
    from regime_detection.axis_series import _build_axis_outputs

    return _build_axis_outputs(
        dates=[ts.date() for ts in close_index],
        raw_labels=raw_labels,
        raw_evidence=raw_evidence,
        risk_rank=VOLATILITY_RISK_RANK,
        deescalation_days_by_label=hysteresis_config.deescalation_days_by_label,
        escalation_days_by_label=hysteresis_config.escalation_days_by_label,
        default_escalation_days=hysteresis_config.default_escalation_days,
        default_deescalation_days=hysteresis_config.default_deescalation_days,
        max_unknown_freeze_days=hysteresis_config.max_unknown_freeze_days,
        required_inputs=[close],
        required_trading_days=VOLATILITY_REQUIRED_TRADING_DAYS,
        max_freshness_days=context.config.data_quality.max_freshness_days,
        min_completeness=context.config.data_quality.min_completeness,
    )
