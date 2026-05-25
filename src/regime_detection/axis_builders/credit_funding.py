from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
    CREDIT_SPREAD_SOURCE_CODE,
    CreditFundingLabel,
    FEDFUNDS_KEY,
    HYG_KEY,
    HY_OAS_KEY,
    IG_OAS_KEY,
    IOER_LEGACY_KEY,
    IORB_KEY,
    LQD_KEY,
    NFCI_KEY,
    SOFR_KEY,
    TLT_KEY,
    build_rule_inputs_by_date as build_credit_funding_rule_inputs_by_date,
    evaluate_rules_with_evidence as evaluate_credit_funding_rules_with_evidence,
)
from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.axis_builders.per_label import build_per_label_axis_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.axis_builders.staleness import (
    _safe_float,
    staleness_for_source,
)
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    CreditFundingOutput,
    DataQuality,
)


def _build_credit_funding_for_spread_source(
    context: MarketContext,
    feature_store: FeatureStore,
    *,
    spread_source: Literal["oas", "proxy"],
) -> dict[date, CreditFundingOutput] | None:
    """Build credit/funding outputs from pre-computed features and hysteresis."""
    features = feature_store.credit_funding
    if features is None:
        return None
    cf_config = context.config.credit_funding
    if cf_config is None:
        return None

    # Resolve the spread-source-specific series + bias-warning code. The
    # §2C rule schema is scale-invariant (percentile + slope only), so the
    # identical pipeline runs on either the real-OAS metric or the
    # TLT-proxy metric (ADR 0007 / implementation decision) — two parallel outputs,
    # never blended into one series or one label.
    if spread_source == "oas":
        hy_spread_63d = features.hy_oas_63d
        ig_spread_63d = features.ig_oas_63d
        hy_spread_percentile_504d = features.hy_oas_percentile_504d
        hy_spread_slope_21d = features.hy_oas_slope_21d
        ig_spread_slope_21d = features.ig_oas_slope_21d
        bias_warning_code = CREDIT_SPREAD_SOURCE_CODE
        evidence_spread_source = "ice_bofa_oas"
    else:  # "proxy"
        hy_spread_63d = features.hy_tr_differential_63d
        ig_spread_63d = features.ig_tr_differential_63d
        hy_spread_percentile_504d = features.hy_tr_differential_percentile_504d
        hy_spread_slope_21d = features.hy_tr_differential_slope_21d
        ig_spread_slope_21d = features.ig_tr_differential_slope_21d
        bias_warning_code = CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE
        evidence_spread_source = "tlt_total_return_differential"

    spy_close = context.spy_ohlcv["close"]
    volatility_features = feature_store.volatility
    realized_vol_pct = volatility_features.realized_vol_percentile_252d
    realized_vol_21d = volatility_features.realized_vol_21d
    nf_features = feature_store.network_fragility
    if nf_features is None:
        avg_corr_pct_series = pd.Series(float("nan"), index=spy_close.index)
        avg_corr_63d_series = pd.Series(float("nan"), index=spy_close.index)
    else:
        avg_corr_pct_series = nf_features.avg_pairwise_corr_percentile_504d
        avg_corr_63d_series = nf_features.avg_pairwise_corr_63d

    # The credit_funding seam guarantees these series exist on the
    # SPY index when feature_store.credit_funding is non-None.
    cross_asset_closes = context.cross_asset_closes or {}
    macro_series = context.macro_series or {}
    hyg_close = cross_asset_closes.get(HYG_KEY)
    lqd_close = cross_asset_closes.get(LQD_KEY)
    tlt_close = cross_asset_closes.get(TLT_KEY)
    nfci_series = macro_series.get(NFCI_KEY)
    hy_oas_series = macro_series.get(HY_OAS_KEY)
    ig_oas_series = macro_series.get(IG_OAS_KEY)

    # Quality-gate primary inputs. Lookback gates on the 504d percentile
    # window — the longest binding cold-start for any rule predicate.
    required_inputs: list[pd.Series] = [
        hy_spread_63d,
        ig_spread_63d,
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
    session_index = spy_close.index
    hyg_staleness_by_date = staleness_for_source(
        source_name=HYG_KEY, series=hyg_close, session_index=session_index
    )
    lqd_staleness_by_date = staleness_for_source(
        source_name=LQD_KEY, series=lqd_close, session_index=session_index
    )
    tlt_staleness_by_date = staleness_for_source(
        source_name=TLT_KEY, series=tlt_close, session_index=session_index
    )
    hy_oas_staleness_by_date = staleness_for_source(
        source_name=HY_OAS_KEY, series=hy_oas_series, session_index=session_index
    )
    ig_oas_staleness_by_date = staleness_for_source(
        source_name=IG_OAS_KEY, series=ig_oas_series, session_index=session_index
    )
    nfci_staleness_by_date = staleness_for_source(
        source_name=NFCI_KEY, series=nfci_series, session_index=session_index
    )
    # Compute staleness from the raw SOFR and FEDFUNDS inputs rather than the
    # already-spliced sofr_iorb_spread. The derived spread is forward-filled in
    # compute_credit_funding_features, so it is always non-NaN and would never
    # detect a real data outage. Taking np.minimum of both raw series preserves
    # ADR 0009: a session covered by the FEDFUNDS-IOER proxy reads its staleness
    # from FEDFUNDS (fresh), not from SOFR (sentinel/stale).
    _sofr_staleness = staleness_for_source(
        source_name=SOFR_KEY,
        series=macro_series.get(SOFR_KEY),
        session_index=session_index,
    )
    _iorb_staleness = staleness_for_source(
        source_name=IORB_KEY,
        series=macro_series.get(IORB_KEY),
        session_index=session_index,
    )
    _fedfunds_staleness = staleness_for_source(
        source_name=FEDFUNDS_KEY,
        series=macro_series.get(FEDFUNDS_KEY),
        session_index=session_index,
    )
    _ioer_legacy_staleness = staleness_for_source(
        source_name=IOER_LEGACY_KEY,
        series=macro_series.get(IOER_LEGACY_KEY),
        session_index=session_index,
    )
    sofr_iorb_pair_staleness = np.maximum(
        _sofr_staleness.to_numpy(), _iorb_staleness.to_numpy()
    )
    sofr_ioer_pair_staleness = np.maximum(
        _sofr_staleness.to_numpy(), _ioer_legacy_staleness.to_numpy()
    )
    fedfunds_ioer_pair_staleness = np.maximum(
        _fedfunds_staleness.to_numpy(), _ioer_legacy_staleness.to_numpy()
    )
    funding_spread_staleness_by_date = pd.Series(
        np.minimum.reduce(
            [
                sofr_iorb_pair_staleness,
                sofr_ioer_pair_staleness,
                fedfunds_ioer_pair_staleness,
            ]
        ),
        index=session_index,
        dtype="int64",
    )
    rule_inputs_by_date = build_credit_funding_rule_inputs_by_date(
        features=features,
        hy_spread_percentile_504d=hy_spread_percentile_504d,
        hy_spread_slope_21d=hy_spread_slope_21d,
        ig_spread_slope_21d=ig_spread_slope_21d,
        realized_vol_21d_percentile_252d=realized_vol_pct,
        avg_pairwise_corr_percentile_504d=avg_corr_pct_series,
        realized_vol_21d=realized_vol_21d,
        avg_pairwise_corr_63d=avg_corr_63d_series,
    )

    for day in context.sessions:
        dt = pd.Timestamp(day)

        # §2C Unknown Gate (HYG/LQD/TLT > 5 sessions, NFCI > 14d,
        # OAS/funding > max_freshness_days). Run BEFORE the generic
        # assess_series_input_quality so the §2C-specific staleness
        # messages reach evidence.
        etf_staleness_breach = False
        etf_stale_label: str | None = None
        hyg_staleness = int(hyg_staleness_by_date.loc[dt])
        lqd_staleness = int(lqd_staleness_by_date.loc[dt])
        tlt_staleness = int(tlt_staleness_by_date.loc[dt])
        if hyg_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "HYG"
        elif lqd_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "LQD"
        elif tlt_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "TLT"

        funding_spread_staleness_days = int(funding_spread_staleness_by_date.loc[dt])
        funding_spread_stale = funding_spread_staleness_days > max_freshness_days
        hy_oas_staleness_days = int(hy_oas_staleness_by_date.loc[dt])
        ig_oas_staleness_days = int(ig_oas_staleness_by_date.loc[dt])
        oas_stale = spread_source == "oas" and (
            hy_oas_staleness_days > max_freshness_days
            or ig_oas_staleness_days > max_freshness_days
        )
        nfci_staleness_days = int(nfci_staleness_by_date.loc[dt])
        nfci_stale = nfci_staleness_days > cf_config.nfci_stale_days

        if etf_staleness_breach or funding_spread_stale or nfci_stale or oas_stale:
            reason_parts: list[str] = []
            if etf_staleness_breach:
                reason_parts.append(f"etf_stale:{etf_stale_label}")
            if oas_stale:
                if hy_oas_staleness_days > max_freshness_days:
                    reason_parts.append(f"hy_oas_stale_{hy_oas_staleness_days}d")
                if ig_oas_staleness_days > max_freshness_days:
                    reason_parts.append(f"ig_oas_stale_{ig_oas_staleness_days}d")
            if funding_spread_stale:
                reason_parts.append(
                    f"funding_spread_stale_{funding_spread_staleness_days}d"
                )
            if nfci_stale:
                reason_parts.append(f"nfci_stale_{nfci_staleness_days}d")
            gate_reason = ",".join(reason_parts)
            raw_labels.append("unknown")
            per_day_data_quality.append(
                DataQuality(
                    status="stale_data",
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

        rule_inputs = rule_inputs_by_date[dt]
        if pd.isna(rule_inputs.hy_spread_percentile_504d):
            reason = "hy_spread_percentile_504d_warmup"
            raw_labels.append("unknown")
            per_day_data_quality.append(
                DataQuality(
                    status="insufficient_history",
                    freshness_days=None,
                    completeness=None,
                    reason=reason,
                )
            )
            per_day_evidence.append(
                {
                    "reason": reason,
                    "spread_source": evidence_spread_source,
                    "bias_warning_code": bias_warning_code,
                }
            )
            continue
        rule_evaluation = evaluate_credit_funding_rules_with_evidence(
            inputs=rule_inputs,
            config=cf_config.rules,
        )
        label = rule_evaluation.label
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append(
            {
                "rule_evidence": {
                    "hy_spread_percentile_504d": rule_inputs.hy_spread_percentile_504d,
                    "hy_spread_slope_21d": rule_inputs.hy_spread_slope_21d,
                    "ig_spread_slope_21d": rule_inputs.ig_spread_slope_21d,
                    "broad_usd_index_zscore_21d": rule_inputs.broad_usd_index_zscore_21d,
                    "sofr_iorb_slope_21d": rule_inputs.sofr_iorb_slope_21d,
                    "spy_21d_return": rule_inputs.spy_21d_return,
                    "tlt_21d_return": rule_inputs.tlt_21d_return,
                    "realized_vol_21d_percentile_252d": rule_inputs.realized_vol_21d_percentile_252d,
                    "realized_vol_21d": rule_inputs.realized_vol_21d,
                    "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
                    "avg_pairwise_corr_63d": rule_inputs.avg_pairwise_corr_63d,
                    "rule_path": rule_evaluation.rule_path,
                    "rule_reason": rule_evaluation.reason,
                },
                "spread_source": evidence_spread_source,
                "nfci_daily_carried": _safe_float(nfci_carried, dt),
                "kre_spy_slope_63d": _safe_float(features.kre_spy_slope_63d, dt),
                "bias_warning_code": bias_warning_code,
            }
        )

    return build_per_label_axis_outputs(
        sessions=context.sessions,
        raw_labels=raw_labels,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=cf_config.deescalation_days_by_label,
        default_deescalation_days=cf_config.default_deescalation_days,
        max_unknown_freeze_days=cf_config.max_unknown_freeze_days,
        data_quality=per_day_data_quality,
        evidence=per_day_evidence,
        output_factory=CreditFundingOutput,
    )


def build_credit_funding_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, CreditFundingOutput] | None:
    """Real-OAS §2C credit/funding labels → ``RegimeOutput.credit_funding_state``."""
    return _build_credit_funding_for_spread_source(
        context, feature_store, spread_source="oas"
    )


def build_credit_funding_proxy_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, CreditFundingOutput] | None:
    """TLT-vs-HYG/LQD proxy §2C labels → ``RegimeOutput.credit_funding_state_proxy``
    (ADR 0007 / implementation decision). Same scale-invariant rule schema, parallel run on
    the proxy series — never blended with the real-OAS labels."""
    return _build_credit_funding_for_spread_source(
        context, feature_store, spread_source="proxy"
    )


def _credit_output_has_classified_signal(output: CreditFundingOutput | None) -> bool:
    if output is None:
        return False
    return output.active_label != "unknown" and output.data_quality.status in {
        "ok",
        "degraded",
    }


def _credit_output_is_data_unavailable(output: CreditFundingOutput | None) -> bool:
    if output is None:
        return True
    return output.classification_status in {
        "data_unavailable",
        "stale_data",
        "insufficient_history",
    }


def _with_effective_credit_evidence(
    *,
    chosen: CreditFundingOutput,
    oas: CreditFundingOutput | None,
    proxy: CreditFundingOutput | None,
    source_used: str,
    agreement_status: str,
) -> CreditFundingOutput:
    evidence = dict(chosen.evidence)
    evidence.update(
        {
            "source_used": source_used,
            "agreement_status": agreement_status,
            "oas_label": oas.active_label if oas is not None else None,
            "proxy_label": proxy.active_label if proxy is not None else None,
            "oas_classification_status": (
                oas.classification_status if oas is not None else None
            ),
            "proxy_classification_status": (
                proxy.classification_status if proxy is not None else None
            ),
            "oas_spread_source": (
                oas.evidence.get("spread_source") if oas is not None else None
            ),
            "proxy_spread_source": (
                proxy.evidence.get("spread_source") if proxy is not None else None
            ),
        }
    )
    return chosen.model_copy(update={"evidence": evidence})


def resolve_credit_funding_effective_output(
    *,
    oas: CreditFundingOutput | None,
    proxy: CreditFundingOutput | None,
) -> CreditFundingOutput | None:
    """Resolve OAS + ETF proxy into the one label consumed downstream.

    OAS and proxy are still emitted separately. The effective output uses OAS
    when it is the only classified signal, uses proxy when OAS is unavailable,
    and uses the higher-risk label when both directional signals disagree.
    """
    if oas is None and proxy is None:
        return None

    oas_signal = _credit_output_has_classified_signal(oas)
    proxy_signal = _credit_output_has_classified_signal(proxy)

    if oas_signal and proxy_signal:
        assert oas is not None and proxy is not None
        oas_rank = CREDIT_FUNDING_RISK_RANK[oas.active_label]
        proxy_rank = CREDIT_FUNDING_RISK_RANK[proxy.active_label]
        if oas_rank == proxy_rank:
            return _with_effective_credit_evidence(
                chosen=oas,
                oas=oas,
                proxy=proxy,
                source_used="oas_confirmed",
                agreement_status="confirmed",
            )
        if proxy_rank > oas_rank:
            return _with_effective_credit_evidence(
                chosen=proxy,
                oas=oas,
                proxy=proxy,
                source_used="proxy_higher_risk",
                agreement_status="divergent",
            )
        return _with_effective_credit_evidence(
            chosen=oas,
            oas=oas,
            proxy=proxy,
            source_used="oas_higher_risk",
            agreement_status="divergent",
        )

    if proxy_signal:
        assert proxy is not None
        return _with_effective_credit_evidence(
            chosen=proxy,
            oas=oas,
            proxy=proxy,
            source_used=(
                "proxy_fallback"
                if _credit_output_is_data_unavailable(oas)
                else "proxy_only"
            ),
            agreement_status="proxy_only",
        )

    if oas_signal:
        assert oas is not None
        return _with_effective_credit_evidence(
            chosen=oas,
            oas=oas,
            proxy=proxy,
            source_used="oas_only",
            agreement_status="oas_only",
        )

    # Neither OAS nor proxy produced a classified label. Prefer the proxy
    # output when it evaluated rules but none fired (no_rule_fired) — this
    # is more honest than forwarding the OAS stale_data status, which
    # misrepresents a rule gap as a data problem.
    if proxy is not None and proxy.classification_status in (
        "no_rule_fired",
        "no_rule_fired_hysteresis",
        "no_rule_fired_missing_feature",
    ):
        chosen = proxy
    else:
        chosen = oas if oas is not None else proxy
    assert chosen is not None
    return _with_effective_credit_evidence(
        chosen=chosen,
        oas=oas,
        proxy=proxy,
        source_used="no_classified_signal",
        agreement_status="unavailable",
    )


def resolve_credit_funding_effective_series(
    *,
    sessions: list[date],
    oas_by_date: dict[date, CreditFundingOutput] | None,
    proxy_by_date: dict[date, CreditFundingOutput] | None,
) -> dict[date, CreditFundingOutput] | None:
    if oas_by_date is None and proxy_by_date is None:
        return None
    outputs: dict[date, CreditFundingOutput] = {}
    for day in sessions:
        effective = resolve_credit_funding_effective_output(
            oas=oas_by_date.get(day) if oas_by_date is not None else None,
            proxy=proxy_by_date.get(day) if proxy_by_date is not None else None,
        )
        if effective is not None:
            outputs[day] = effective
    return outputs
