from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING


from regime_detection.breadth_state import (
    _RISK_RANK as BREADTH_RISK_RANK,
    _data_quality_for_asof as breadth_data_quality_for_asof,
    build_raw_outputs as build_breadth_raw_outputs,
    resolve_v2_raw_outputs as resolve_breadth_v2_raw_outputs,
)
from regime_detection.data_quality import quality_forces_unknown
from regime_detection.feature_store import FeatureStore
from regime_detection.hysteresis import (
    apply_asymmetric_hysteresis,
)
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    BreadthStateOutput,
    DataQuality,
)

if TYPE_CHECKING:
    from regime_detection.axis_series import AxisSeriesResult


_PIT_BREADTH_LABELS = {
    "breadth_thrust",
    "narrowing_breadth",
    "recovery_breadth",
    "broadening_breadth",
}
_STALENESS_SENTINEL = 10**9

# V1 breadth ETF-proxy quality gate uses the existing 50-session calibration.
BREADTH_REQUIRED_TRADING_DAYS = 50


def build_breadth_axis_series(
    context: MarketContext, feature_store: FeatureStore
) -> AxisSeriesResult:
    spy_close = context.spy_ohlcv["close"]
    rsp_close = context.rsp_close.reindex(context.spy_ohlcv.index)
    features = feature_store.breadth
    raw_labels, raw_evidence = build_breadth_raw_outputs(features)

    # V2 §1D extension (documented implementation decision): when the PIT seam is
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
        raw_labels, raw_evidence = resolve_breadth_v2_raw_outputs(
            dates=spy_close.index,
            raw_labels=raw_labels,
            raw_evidence=raw_evidence,
            pct_above_50dma=v2_features.pct_above_50dma,
            pct_above_200dma=v2_features.pct_above_200dma,
            nh_nl_ratio=v2_features.nh_nl_ratio,
            ad_line_slope_20d=v2_features.ad_line_slope_20d,
            breadth_thrust=v2_features.breadth_thrust,
            lookback_sessions=v2_config.label_rate_of_change_lookback_sessions,
            nh_nl_threshold=v2_config.nh_nl_ratio_narrowing_threshold,
        )

    stable_labels, active_labels = apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=BREADTH_RISK_RANK,
        escalation_days=context.config.hysteresis.breadth_escalation_days,
        deescalation_days=context.config.hysteresis.breadth_deescalation_days,
    )
    outputs_by_date: dict[date, BreadthStateOutput] = {}
    stable_by_date: dict[date, str] = {}
    active_by_date: dict[date, str] = {}
    for day, raw, stable, active, evidence in zip(
        spy_close.index.date,
        raw_labels,
        stable_labels,
        active_labels,
        raw_evidence,
        strict=True,
    ):
        mode = (
            "pit_constituent_biased_research"
            if {raw, stable, active} & _PIT_BREADTH_LABELS
            else "etf_proxy"
        )
        if raw == "unknown":
            output = BreadthStateOutput(
                mode=mode,
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
            data_quality = breadth_data_quality_for_asof(
                spy_close=spy_close,
                rsp_close=rsp_close,
                as_of_date=day,
                required_trading_days=BREADTH_REQUIRED_TRADING_DAYS,
                raw_label=raw,
                max_freshness_days=context.config.data_quality.max_freshness_days,
                min_completeness=context.config.data_quality.min_completeness,
            )
            if quality_forces_unknown(data_quality):
                output = BreadthStateOutput(
                    mode=mode,
                    raw_label="unknown",
                    stable_label="unknown",
                    active_label="unknown",
                    evidence={"reason": data_quality.reason, "proxy": "RSP/SPY"},
                    data_quality=data_quality,
                )
            else:
                output = BreadthStateOutput(
                    mode=mode,
                    raw_label=raw,
                    stable_label=stable,
                    active_label=active,
                    evidence={
                        "proxy": "RSP/SPY",
                        "rule_evidence": evidence,
                        "risk_rank": BREADTH_RISK_RANK,
                        "deescalation_days": context.config.hysteresis.breadth_deescalation_days,
                    },
                    data_quality=data_quality,
                )
        outputs_by_date[day] = output
        stable_by_date[day] = output.stable_label
        active_by_date[day] = output.active_label
    from regime_detection.axis_series import AxisSeriesResult

    return AxisSeriesResult(
        outputs_by_date=outputs_by_date,
        stable_labels_by_date=stable_by_date,
        active_labels_by_date=active_by_date,
    )
