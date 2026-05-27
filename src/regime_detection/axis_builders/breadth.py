from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

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
from regime_detection.hysteresis import apply_data_quality_aware_hysteresis
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
# V1 breadth ETF-proxy quality gate uses the existing 50-session calibration.
BREADTH_REQUIRED_TRADING_DAYS = 50


def _derive_breadth_active_label_source(
    *,
    raw: str,
    stable: str,
    active: str,
) -> str:
    if active != raw:
        return "hysteresis_from_prior_state"
    if active in _PIT_BREADTH_LABELS:
        return "pit_constituent"
    return "etf_proxy"


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
    # apply the §1D precedence walk (spec line 385). When the PIT seam is
    # unlit (default-config callers, no PIT inputs), V2 rules silently do
    # NOT fire — V1 byte-identity is preserved (AGENTS.md V1 archive
    # replay rule; tests/test_v1_frozen_replay.py).
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

    hysteresis_config = context.config.breadth_state
    per_day_data_quality: list[DataQuality] = []
    for day, raw in zip(spy_close.index.date, raw_labels, strict=True):
        if raw == "unknown":
            per_day_data_quality.append(
                DataQuality(
                    status="insufficient_history",
                    freshness_days=None,
                    completeness=None,
                    reason="required_feature_is_nan",
                )
            )
            continue
        per_day_data_quality.append(
            breadth_data_quality_for_asof(
                spy_close=spy_close,
                rsp_close=rsp_close,
                as_of_date=day,
                required_trading_days=BREADTH_REQUIRED_TRADING_DAYS,
                raw_label=raw,
                max_freshness_days=context.config.data_quality.max_freshness_days,
                min_completeness=context.config.data_quality.min_completeness,
            )
        )
    stable_labels, active_labels, frozen_labels = apply_data_quality_aware_hysteresis(
        raw_labels=raw_labels,
        risk_rank=BREADTH_RISK_RANK,
        deescalation_days_by_label=hysteresis_config.deescalation_days_by_label,
        default_deescalation_days=hysteresis_config.default_deescalation_days,
        max_unknown_freeze_days=hysteresis_config.max_unknown_freeze_days,
        data_quality=per_day_data_quality,
    )
    outputs_by_date: dict[date, BreadthStateOutput] = {}
    stable_by_date: dict[date, str] = {}
    active_by_date: dict[date, str] = {}
    for day, raw, stable, active, is_frozen, evidence, data_quality in zip(
        spy_close.index.date,
        raw_labels,
        stable_labels,
        active_labels,
        frozen_labels,
        raw_evidence,
        per_day_data_quality,
        strict=True,
    ):
        mode = (
            "pit_constituent_biased_research"
            if {raw, stable, active} & _PIT_BREADTH_LABELS
            else "etf_proxy"
        )
        active_label_source = _derive_breadth_active_label_source(
            raw=raw,
            stable=stable,
            active=active,
        )
        if is_frozen:
            output = BreadthStateOutput(
                mode=mode,
                raw_label="unknown",
                stable_label=stable,
                active_label=active,
                evidence={
                    "reason": data_quality.reason,
                    "proxy": "RSP/SPY",
                    "row_provenance_mode": mode,
                    "active_label_source": "data_quality_freeze",
                    "data_quality_freeze": True,
                },
                data_quality=data_quality,
            )
        elif quality_forces_unknown(data_quality):
            output = BreadthStateOutput(
                mode=mode,
                raw_label=raw,
                stable_label="unknown",
                active_label="unknown",
                evidence={
                    "reason": data_quality.reason,
                    "proxy": "RSP/SPY",
                    "row_provenance_mode": mode,
                    "active_label_source": active_label_source,
                },
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
                    "deescalation_days": hysteresis_config.default_deescalation_days,
                    "row_provenance_mode": mode,
                    "active_label_source": active_label_source,
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
