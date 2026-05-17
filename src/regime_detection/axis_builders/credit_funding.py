from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
    CREDIT_SPREAD_SOURCE_CODE,
    CreditFundingLabel,
    build_rule_inputs_by_date as build_credit_funding_rule_inputs_by_date,
    evaluate_rules as evaluate_credit_funding_rules,
)
from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.axis_builders.per_label import build_per_label_axis_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.axis_builders.staleness import (
    _calendar_staleness_days_series,
    _safe_float,
    _trading_staleness_series,
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
    """V2 §2C credit/funding axis classifier (implementation phase).

    Pipeline:

      1. Read pre-computed features from ``feature_store.credit_funding``
         (compute_credit_funding_features). If the seam is None (no v2
         config / required inputs absent) return None — the timeline then
         leaves ``RegimeOutput.credit_funding_state`` as ``None``.
      2. Per session, run the §2C unknown gate (spec lines 2122-2126):
         - HYG/LQD/TLT stale > 5 sessions → unknown
         - NFCI stale > 14 days → unknown
         - SOFR or IORB stale beyond the global freshness budget → unknown
         - assess_series_input_quality fails on any required series → unknown
      3. Materialize per-day scalar rule inputs (build_rule_inputs_for_date),
         then evaluate §2C precedence (deleveraging > funding_squeeze >
         credit_stress > spread_widening > credit_calm > unknown).
      4. Apply per-label asymmetric hysteresis (§2C lines 2105-2118).
      5. Emit one CreditFundingOutput per session.
    """
    features = feature_store.credit_funding
    if features is None:
        return None
    cf_config = context.config.credit_funding
    if cf_config is None:
        return None

    # Resolve the spread-source-specific series + bias-warning code. The
    # §2C rule schema is scale-invariant (percentile + slope only), so the
    # identical pipeline runs on either the real-OAS metric or the
    # TLT-proxy metric (documented implementation decision) — two parallel outputs, never
    # blended into one series or one label.
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
    hyg_staleness_by_date = _trading_staleness_series(hyg_close, session_index)
    lqd_staleness_by_date = _trading_staleness_series(lqd_close, session_index)
    tlt_staleness_by_date = _trading_staleness_series(tlt_close, session_index)
    nfci_staleness_by_date = _calendar_staleness_days_series(nfci_series, session_index)
    sofr_staleness_by_date = _calendar_staleness_days_series(sofr_series, session_index)
    iorb_staleness_by_date = _calendar_staleness_days_series(iorb_series, session_index)
    rule_inputs_by_date = build_credit_funding_rule_inputs_by_date(
        features=features,
        hy_spread_percentile_504d=hy_spread_percentile_504d,
        hy_spread_slope_21d=hy_spread_slope_21d,
        ig_spread_slope_21d=ig_spread_slope_21d,
        realized_vol_21d_percentile_252d=realized_vol_pct,
        avg_pairwise_corr_percentile_504d=avg_corr_pct_series,
    )

    for day in context.sessions:
        dt = pd.Timestamp(day)

        # §2C unknown gate (spec lines 2122-2126). Run BEFORE the generic
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

        sofr_staleness_days = int(sofr_staleness_by_date.loc[dt])
        iorb_staleness_days = int(iorb_staleness_by_date.loc[dt])
        sofr_stale = sofr_staleness_days > max_freshness_days
        iorb_stale = iorb_staleness_days > max_freshness_days
        nfci_staleness_days = int(nfci_staleness_by_date.loc[dt])
        nfci_stale = nfci_staleness_days > cf_config.nfci_stale_days

        if etf_staleness_breach or sofr_stale or iorb_stale or nfci_stale:
            reason_parts: list[str] = []
            if etf_staleness_breach:
                reason_parts.append(f"etf_stale:{etf_stale_label}")
            if sofr_stale:
                reason_parts.append(f"sofr_stale_{sofr_staleness_days}d")
            if iorb_stale:
                reason_parts.append(f"iorb_stale_{iorb_staleness_days}d")
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

        rule_inputs = rule_inputs_by_date[dt]
        label = evaluate_credit_funding_rules(
            inputs=rule_inputs,
            config=cf_config.rules,
        )
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
                    "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
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
    (documented implementation decision). Same scale-invariant rule schema, parallel run on
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
