from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.feature_store import FeatureStore
from regime_detection.hysteresis import (
    apply_per_label_asymmetric_hysteresis,
)
from regime_detection.market_context import MarketContext
from regime_detection.models import (
    DataQuality,
    NetworkFragilityOutput,
)
from regime_detection.network_fragility_rules import (
    NETWORK_FRAGILITY_RISK_RANK,
    NetworkFragilityLabel,
    build_rule_inputs_by_date,
    evaluate_rules,
)


def _assess_network_fragility_day_quality(
    *,
    day: date,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
) -> DataQuality:
    return assess_series_input_quality(
        as_of_date=day,
        required_inputs=required_inputs,
        required_trading_days=required_trading_days,
        raw_label="",
        max_freshness_days=max_freshness_days,
        min_completeness=min_completeness,
        skip_raw_label_short_circuit=True,
    )


def build_network_fragility_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
    breadth_active_labels_by_date: dict[date, str] | None = None,
    volatility_active_labels_by_date: dict[date, str] | None = None,
    credit_funding_active_labels_by_date: dict[date, str] | None = None,
) -> dict[date, NetworkFragilityOutput] | None:
    """V2 §3 network fragility classifier — full pipeline (Slice 1.4).

    Pipeline per v2 spec §3.2–§3.7:

      1. Read pre-computed features from ``feature_store.network_fragility``
         (compute_features → v2 §3.2). If the seam is None (no sector ETF
         data) return None so the timeline falls back to the v2 "unknown"
         placeholder shape.
      2. For each session date, materialize the per-day scalar rule inputs
         (build_rule_inputs_for_date) and assess data quality
         (assess_series_input_quality + quality_forces_unknown). Quality
         failures override the rule output with "unknown".
      3. Cross-reference V1 axes (breadth_state.active_label,
         volatility_state.active_label) for that date.
      4. Evaluate rules (evaluate_rules) → raw label per v2 §3.4 precedence.
         credit_funding_label is read from the authoritative §2C axis when
         supplied; otherwise it stays None and systemic_stress short-circuits
         to False per the spec.
      5. Apply per-label asymmetric hysteresis (v2 §3.7) over the raw label
         series.
      6. Emit one NetworkFragilityOutput per session.
    """
    features = feature_store.network_fragility
    if features is None:
        return None

    network_fragility_config = context.config.network_fragility
    if network_fragility_config is None:
        # Defensive: the feature seam is populated but config is missing.
        # Treat as no v2 axis available rather than crashing the engine.
        return None

    spy_close = context.spy_ohlcv["close"]
    volatility_features = feature_store.volatility
    realized_vol_pct = volatility_features.realized_vol_percentile_252d
    vix_pct = (
        volatility_features.vix_percentile_252d
        if volatility_features.vix_percentile_252d is not None
        else pd.Series(float("nan"), index=spy_close.index)
    )

    # Required-input check (v2 §2.8 data quality).
    # The percentile features are derived and are structurally NaN until
    # the 504d / 252d windows fill — the rules themselves return False on
    # NaN inputs (falling through to "unknown" per v2 §3.3), so we gate
    # on the underlying primary inputs only and let day-level NaN
    # propagate through the rule engine.
    required_inputs: list[pd.Series] = [
        features.avg_pairwise_corr_63d,
        features.largest_eigenvalue_share,
        features.effective_rank,
        features.dispersion_ratio,
        spy_close,
    ]
    # The 63d correlation window is the longest mandatory raw-input
    # lookback (v2 §3.2 line 554-558). The 504d percentile / 21d slope
    # NaN-ness is handled inside the rule predicates per spec.
    required_trading_days = network_fragility_config.correlation_lookback_days
    max_freshness_days = context.config.data_quality.max_freshness_days
    min_completeness = context.config.data_quality.min_completeness

    raw_labels: list[NetworkFragilityLabel] = []
    per_day_data_quality: list[DataQuality] = []
    per_day_evidence: list[dict[str, object]] = []
    rule_inputs_by_date = build_rule_inputs_by_date(
        features=features,
        spy_close=spy_close,
        realized_vol_percentile_252d=realized_vol_pct,
        vix_percentile_252d=vix_pct,
    )

    for day in context.sessions:
        dt = pd.Timestamp(day)

        # Defensive: the volatility series may not include early sessions
        # if cold-start clipping diverges. Reindex on the fly.
        # Pure-quality assessment: we compute the raw label AFTER quality
        # (rule engine, below) and re-check with quality_forces_unknown.
        # `skip_raw_label_short_circuit=True` keeps the helper from
        # collapsing to "insufficient_history" before we have a label.
        day_quality = _assess_network_fragility_day_quality(
            day=day,
            required_inputs=required_inputs,
            required_trading_days=required_trading_days,
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

        # I1: strict V1 axis alignment. When the caller supplied a v1
        # dict, missing-day → KeyError (loud) rather than silent
        # "unknown" substitution which would defang systemic_stress on
        # any drifted session. The "unknown" fallback applies ONLY when
        # the caller explicitly omitted the v1 dict (unit-test path).
        if breadth_active_labels_by_date is None:
            breadth_label = "unknown"
        else:
            if day not in breadth_active_labels_by_date:
                raise KeyError(
                    f"breadth_active_labels_by_date missing session {day!r} "
                    "(v1/v2 calendar drift would silently downgrade rules to 'unknown')"
                )
            breadth_label = breadth_active_labels_by_date[day]
        if volatility_active_labels_by_date is None:
            volatility_label = "unknown"
        else:
            if day not in volatility_active_labels_by_date:
                raise KeyError(
                    f"volatility_active_labels_by_date missing session {day!r} "
                    "(v1/v2 calendar drift would silently downgrade rules to 'unknown')"
                )
            volatility_label = volatility_active_labels_by_date[day]
        credit_funding_label: str | None = None
        if credit_funding_active_labels_by_date is not None:
            if day not in credit_funding_active_labels_by_date:
                raise KeyError(
                    f"credit_funding_active_labels_by_date missing session {day!r} "
                    "(v2/v2 calendar drift would silently downgrade systemic_stress)"
                )
            credit_funding_label = credit_funding_active_labels_by_date[day]

        label = evaluate_rules(
            inputs=rule_inputs,
            config=network_fragility_config.rules,
            breadth_label=breadth_label,  # type: ignore[arg-type]
            volatility_label=volatility_label,  # type: ignore[arg-type]
            credit_funding_label=credit_funding_label,  # type: ignore[arg-type]
        )
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append(
            {
                "rule_evidence": {
                    "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
                    "largest_eigenvalue_share_percentile_504d": rule_inputs.largest_eigenvalue_share_percentile_504d,
                    "effective_rank_percentile_504d": rule_inputs.effective_rank_percentile_504d,
                    "dispersion_ratio_percentile_252d": rule_inputs.dispersion_ratio_percentile_252d,
                    "realized_vol_percentile_252d": rule_inputs.realized_vol_percentile_252d,
                    "drawdown_21d": rule_inputs.drawdown_21d,
                    "vix_percentile_252d": rule_inputs.vix_percentile_252d,
                },
                "breadth_active_label": breadth_label,
                "volatility_active_label": volatility_label,
                "credit_funding_active_label": credit_funding_label,
            }
        )

    stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=NETWORK_FRAGILITY_RISK_RANK,
        deescalation_days_by_label=network_fragility_config.deescalation_days_by_label,
        default_deescalation_days=network_fragility_config.default_deescalation_days,
    )

    outputs: dict[date, NetworkFragilityOutput] = {}
    for day, raw, stable, active, dq, evidence in zip(
        context.sessions,
        raw_labels,
        stable_labels,
        active_labels,
        per_day_data_quality,
        per_day_evidence,
        strict=True,
    ):
        outputs[day] = NetworkFragilityOutput(
            raw_label=raw,
            stable_label=stable,
            active_label=active,
            evidence=evidence,
            data_quality=dq,
        )
    return outputs
