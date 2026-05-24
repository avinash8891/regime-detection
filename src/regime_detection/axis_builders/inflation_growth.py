from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.axis_builders.per_label import build_per_label_axis_outputs
from regime_detection.feature_store import FeatureStore
from regime_detection.axis_builders.staleness import (
    _calendar_staleness_days_series,
    _trading_staleness_series,
)
from regime_detection.inflation_growth import (
    CPI_KEY,
    DGS10_KEY,
    INFLATION_GROWTH_RISK_RANK,
    InflationGrowthLabel,
    PMI_KEY,
    build_rule_inputs_by_date as build_inflation_growth_rule_inputs_by_date,
    evaluate_rules as evaluate_inflation_growth_rules,
)

from regime_detection.market_context import MarketContext
from regime_detection.models import (
    DataQuality,
    InflationGrowthOutput,
)

_EPS_REVISION_MACRO_KEY = "aggregate_forward_eps_revision"
_CPI_NOWCAST_MACRO_KEY = "cpi_nowcast"


def _cpi_staleness_source(
    latest_cpi: pd.Series | None,
    first_release_cpi: pd.Series | None,
    *,
    use_first_release: bool,
) -> pd.Series | None:
    """Produce the CPI **release-timestamp** source for staleness checks.

    Returns the union of observation timestamps across both vintages (so the
    "calendar days since most recent CPI release" measurement sees every
    release, not just one vintage). Values are irrelevant — only the index
    matters here; that's why this does *not* reindex to a session calendar
    or ffill, unlike the value-series helper in
    ``regime_detection.inflation_growth._cpi_with_first_release_fallback``.
    """
    if first_release_cpi is None or not use_first_release:
        return latest_cpi
    if latest_cpi is None:
        return first_release_cpi
    return pd.concat([latest_cpi, first_release_cpi]).sort_index()


def build_inflation_growth_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
    credit_funding_active_labels_by_date: dict[date, str] | None = None,
) -> dict[date, InflationGrowthOutput] | None:
    """V2 §2B inflation/growth axis classifier.

    Pipeline:

      1. Read pre-computed features from ``feature_store.inflation_growth``.
         If the seam is None (no v2 config / required inputs absent) return
         None — the timeline leaves ``RegimeOutput.inflation_growth_state``
         as None.
      2. Per session, run the §2B Unknown Gate (spec ~lines 3151-3155):
         - CPI series stale > 60 calendar days
         - PMI series stale > 45 calendar days
         - DGS10 stale > 5 sessions
         - ``assess_series_input_quality`` fails on any required series
      3. Cross-thread the §2C credit_funding.active_label for the session;
         None falls through to the §2B Cross-Axis Short-Circuit (spec
         ~lines 3157-3161) which falsifies goldilocks / recession_scare /
         recovery_growth.
      4. Materialize per-day scalar rule inputs and evaluate §2B precedence
         (inflation_shock > recession_scare > risk_off_mild > disinflation >
         goldilocks > recovery_growth > reflation > stagflation_lite >
         earnings_contraction > earnings_expansion > unknown). Optional
         nowcast/EPS inputs falsify via NaN until their source series are
         present.
      5. Apply per-label asymmetric hysteresis (§2B Hysteresis, ~spec lines 3128-3144).
      6. Emit one InflationGrowthOutput per session.
    """
    features = feature_store.inflation_growth
    if features is None:
        return None
    ig_config = context.config.inflation_growth
    if ig_config is None:
        return None

    spy_close = context.spy_ohlcv["close"]
    macro_series = context.macro_series or {}
    cpi_series = macro_series.get(CPI_KEY)
    cpi_staleness_series = _cpi_staleness_source(
        cpi_series,
        context.cpi_first_release,
        use_first_release=ig_config.rules.use_first_release_cpi_when_available,
    )
    pmi_series = macro_series.get(PMI_KEY)
    dgs10_series = macro_series.get(DGS10_KEY)
    nowcast_series = macro_series.get(_CPI_NOWCAST_MACRO_KEY)
    eps_revision_series = macro_series.get(_EPS_REVISION_MACRO_KEY)

    # The 126d (6m) CPI lookback is the binding cold-start window.
    required_inputs: list[pd.Series] = [
        features.cpi_6m_change_pct,
        features.pmi_manufacturing,
        features.treasury_10y_yield_slope_21d,
        features.commodity_return_63d,
        spy_close,
    ]
    required_trading_days = ig_config.rules.cpi_lookback_6m_sessions
    max_freshness_days = context.config.data_quality.max_freshness_days
    min_completeness = context.config.data_quality.min_completeness

    raw_labels: list[InflationGrowthLabel] = []
    per_day_data_quality: list[DataQuality] = []
    per_day_evidence: list[dict[str, object]] = []
    session_index = spy_close.index
    cpi_staleness_by_date = _calendar_staleness_days_series(
        cpi_staleness_series, session_index
    )
    pmi_staleness_by_date = _calendar_staleness_days_series(pmi_series, session_index)
    dgs10_staleness_by_date = _trading_staleness_series(dgs10_series, session_index)
    nowcast_staleness_by_date = _calendar_staleness_days_series(
        nowcast_series, session_index
    )
    eps_staleness_by_date = _calendar_staleness_days_series(
        eps_revision_series, session_index
    )
    credit_funding_labels_by_ts: dict[pd.Timestamp, str | None] | None = None
    if credit_funding_active_labels_by_date is not None:
        credit_funding_labels_by_ts = {
            pd.Timestamp(day): label
            for day, label in credit_funding_active_labels_by_date.items()
        }
    rule_inputs_by_date = build_inflation_growth_rule_inputs_by_date(
        features=features,
        config=ig_config.rules,
        credit_funding_active_labels_by_date=credit_funding_labels_by_ts,
    )

    for day in context.sessions:
        dt = pd.Timestamp(day)

        # §2B Unknown Gate (spec ~lines 3151-3155). Run BEFORE the generic
        # quality assessment so the §2B-specific staleness messages reach
        # evidence.
        cpi_staleness_days = int(cpi_staleness_by_date.loc[dt])
        pmi_staleness_days = int(pmi_staleness_by_date.loc[dt])
        dgs10_staleness_sessions = int(dgs10_staleness_by_date.loc[dt])
        cpi_stale = cpi_staleness_days > ig_config.cpi_stale_calendar_days
        pmi_stale = pmi_staleness_days > ig_config.pmi_stale_calendar_days
        dgs10_stale = dgs10_staleness_sessions > ig_config.dgs10_stale_sessions

        if cpi_stale or pmi_stale or dgs10_stale:
            reason_parts: list[str] = []
            if cpi_stale:
                reason_parts.append(f"cpi_stale_{cpi_staleness_days}d")
            if pmi_stale:
                reason_parts.append(f"pmi_stale_{pmi_staleness_days}d")
            if dgs10_stale:
                reason_parts.append(f"dgs10_stale_{dgs10_staleness_sessions}s")
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

        # Cross-axis dependency — None signals the §2C axis is unbuilt,
        # falsifying the goldilocks / recession_scare / recovery_growth
        # predicates per §2B Cross-Axis Short-Circuit (spec ~lines 3157-3161).
        credit_funding_active_label: str | None = None
        if credit_funding_active_labels_by_date is not None:
            if day not in credit_funding_active_labels_by_date:
                raise KeyError(
                    f"credit_funding_active_labels_by_date missing session {day!r} "
                    "(v1/v2 calendar drift would silently downgrade §2B cross-axis rules)"
                )
            credit_funding_active_label = credit_funding_active_labels_by_date[day]

        rule_inputs = rule_inputs_by_date[dt]

        nowcast_staleness_days = int(nowcast_staleness_by_date.loc[dt])
        nowcast_stale = (
            nowcast_series is not None
            and nowcast_staleness_days > ig_config.nowcast_stale_calendar_days
        )
        if nowcast_stale:
            rule_inputs = replace(
                rule_inputs,
                inflation_surprise_zscore=float("nan"),
            )

        # EPS staleness gate — same pattern as the staleness-based unknown
        # gates elsewhere in this module.
        # When the aggregate_forward_eps_revision series has no non-NaN value
        # within eps_revision_stale_calendar_days of this session, treat the
        # EPS direction signal as NaN so earnings_expansion /
        # earnings_contraction predicates falsify rather than forward-filling
        # a stale revision direction.
        eps_staleness_days = int(eps_staleness_by_date.loc[dt])
        eps_stale = eps_staleness_days > ig_config.eps_revision_stale_calendar_days
        if eps_stale:
            rule_inputs = replace(
                rule_inputs,
                aggregate_forward_eps_revision_direction_4w=float("nan"),
            )

        label = evaluate_inflation_growth_rules(
            inputs=rule_inputs, config=ig_config.rules
        )
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append(
            {
                "rule_evidence": {
                    "cpi_6m_change_pct": rule_inputs.cpi_6m_change_pct,
                    "cpi_6m_change_pct_lag_21": rule_inputs.cpi_6m_change_pct_lag_21,
                    "cpi_6m_change_pct_slope_21d": rule_inputs.cpi_6m_change_pct_slope_21d,
                    "inflation_surprise_zscore": rule_inputs.inflation_surprise_zscore,
                    "pmi_manufacturing": rule_inputs.pmi_manufacturing,
                    "pmi_manufacturing_slope_21d": rule_inputs.pmi_manufacturing_slope_21d,
                    "aggregate_forward_eps_revision_direction_4w": rule_inputs.aggregate_forward_eps_revision_direction_4w,
                    "commodity_return_63d": rule_inputs.commodity_return_63d,
                    "treasury_10y_yield_slope_21d": rule_inputs.treasury_10y_yield_slope_21d,
                    "cyclical_defensive_slope_21d": rule_inputs.cyclical_defensive_slope_21d,
                    "spy_21d_return": rule_inputs.spy_21d_return,
                    "tlt_21d_return": rule_inputs.tlt_21d_return,
                },
                "credit_funding_active_label": credit_funding_active_label,
                "bias_warning_code": "commodity_proxy_dbc_substitute",
            }
        )

    return build_per_label_axis_outputs(
        sessions=context.sessions,
        raw_labels=raw_labels,
        risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=ig_config.deescalation_days_by_label,
        default_deescalation_days=ig_config.default_deescalation_days,
        data_quality=per_day_data_quality,
        evidence=per_day_evidence,
        output_factory=InflationGrowthOutput,
    )
