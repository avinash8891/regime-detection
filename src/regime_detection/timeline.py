from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.axis_series import build_axis_series_bundle
from regime_detection.cohort_routing import evaluate_cohort_routing
from regime_detection.config import RegimeConfig
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import MarketContext, slice_context_to_recent_sessions
from regime_detection.models import (
    BreadthStateOutput,
    ChangePointOutput,
    ClusterOutput,
    DataQuality,
    MonetaryPressureOutput,
    NetworkFragilityOutput,
    RegimeOutput,
    RegimeTimeline,
    StructuralCausalState,
)
from regime_detection.strategy_family_constraints import (
    resolve_strategy_family_constraints,
)
from regime_detection.strategy_response import build_strategy_response
from regime_detection.transition_risk_series import build_transition_risk_series
from regime_detection.versioning import engine_version


ENGINE_MINIMUM_HISTORY = 320


def _v2_classifier_not_yet_implemented_data_quality() -> DataQuality:
    """DataQuality for V2 axes whose classifier hasn't shipped yet.

    Mirrors V1 §2.7 NaN cold-start contract (status=insufficient_history,
    reason=required_feature_is_nan, freshness/completeness null).
    """
    return DataQuality(
        status="insufficient_history",
        freshness_days=None,
        completeness=None,
        reason="required_feature_is_nan",
    )


def _resolve_network_fragility_by_date(
    *,
    bundle_entry: dict[date, NetworkFragilityOutput] | None,
    sessions,
) -> dict[date, NetworkFragilityOutput]:
    """Per-day fragility outputs.

    Prefer the AxisSeriesBundle entry when present (slice 1+ supplies real
    classifications). Fall back to a v2 'unknown' placeholder per session
    when sector ETF data wasn't passed and the bundle entry is None.
    """
    if bundle_entry is not None:
        return bundle_entry
    placeholder_dq = _v2_classifier_not_yet_implemented_data_quality()
    return {
        day: NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"reason": "v2_classifier_not_yet_implemented"},
            data_quality=placeholder_dq,
        )
        for day in sessions
    }


def build_regime_timeline(
    *,
    context: MarketContext,
    lookback_days: int,
    config: RegimeConfig | None = None,
) -> RegimeTimeline:
    cfg = config if config is not None else context.config
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0. Got: {lookback_days}")
    if len(context.sessions) < lookback_days:
        raise ValueError(
            "Insufficient NYSE trading-day coverage for requested lookback_days. "
            f"Requested={lookback_days}, available={len(context.sessions)}, "
            f"end_date={context.end_date.isoformat()}."
        )

    # Slice 6/7/8 trainable v2 evidence layers (HMM, GMM clustering, BOCPD
    # change-point) each need the trailing ``training_window_days`` rows of
    # their inputs to fit. Extend the engine's minimum slicing window to
    # the LARGEST configured training window. Without this, disabling one
    # seam (e.g. change_point) but keeping another (e.g. HMM) would slice
    # the context down below HMM's training_window_days and the HMM seam
    # would silently return None for insufficient history.
    # Keeps V1 byte-identity for callers that omit all three configs
    # (max() collapses to ENGINE_MINIMUM_HISTORY).
    v2_min_history = ENGINE_MINIMUM_HISTORY
    trailing_component_lookback = 0
    if cfg.change_point is not None:
        # +21 absorbs the realized_vol_21d warmup so BOCPD sees a full
        # non-NaN training window on the trailing slice.
        v2_min_history = max(
            v2_min_history, cfg.change_point.training_window_days + 21
        )
    if cfg.hmm is not None:
        # +63 absorbs the deepest HMM input warmup (drawdown_63d /
        # avg_pairwise_corr_63d) so the trailing slice gives the GaussianHMM
        # fit a full non-NaN training window.
        v2_min_history = max(
            v2_min_history, cfg.hmm.training_window_days + 63
        )
        # transition_score.hmm_probability_shift needs top_state_prob[t-5].
        # Keep five extra warmed sessions ahead of the emitted window so the
        # first requested output can use the same PIT evidence as later rows.
        trailing_component_lookback = max(trailing_component_lookback, 5)
    if cfg.clustering is not None:
        # +63 absorbs the deepest GMM input warmup (return_63d /
        # drawdown_63d / avg_pairwise_corr_63d) so the trailing slice gives
        # the GaussianMixture fit a full non-NaN training window.
        v2_min_history = max(
            v2_min_history, cfg.clustering.training_window_days + 63
        )
    required_sessions = min(
        len(context.sessions),
        v2_min_history + lookback_days - 1 + trailing_component_lookback,
    )
    working_context = slice_context_to_recent_sessions(context=context, required_sessions=required_sessions)
    network_fragility_config = cfg.network_fragility
    trend_direction_v2_config = cfg.trend_direction_v2
    volatility_state_v2_config = cfg.volatility_state_v2
    breadth_state_v2_config = cfg.breadth_state_v2
    volume_liquidity_v2_config = cfg.volume_liquidity_v2
    monetary_pressure_v2_config = cfg.monetary_pressure_v2
    credit_funding_config = cfg.credit_funding
    inflation_growth_config = cfg.inflation_growth
    central_bank_text_config = cfg.central_bank_text
    news_sentiment_config = cfg.news_sentiment
    feature_store = build_feature_store(
        working_context,
        network_fragility_config=network_fragility_config,
        trend_direction_v2_config=trend_direction_v2_config,
        volatility_state_v2_config=volatility_state_v2_config,
        breadth_state_v2_config=breadth_state_v2_config,
        volume_liquidity_v2_config=volume_liquidity_v2_config,
        monetary_pressure_v2_config=monetary_pressure_v2_config,
        credit_funding_config=credit_funding_config,
        inflation_growth_config=inflation_growth_config,
        central_bank_text_config=central_bank_text_config,
        news_sentiment_config=news_sentiment_config,
    )
    axis_bundle = build_axis_series_bundle(context=working_context, feature_store=feature_store)
    transition_risk = build_transition_risk_series(
        context=working_context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
    )

    selected_days = list(working_context.sessions[-lookback_days:])

    # v2 §6.2 GMM clustering evidence (Slice 7) — bulk-reindex BEFORE the
    # per-day loop (matches the `_build_transition_score_inputs_by_date`
    # pattern). Per-day `.get(pd.Timestamp(day))` would re-scan the index
    # n_sessions times; one reindex + positional access keeps the loop O(N).
    # v2 §6.3 BOCPD change-point evidence (Slice 8) — bulk-reindex BEFORE the
    # per-day loop, same pattern as clustering. Stays None when the seam is
    # absent (config off, or SPY history shorter than training_window_days).
    change_point_features = feature_store.change_point
    if change_point_features is not None:
        cp_session_index = pd.DatetimeIndex(
            [pd.Timestamp(d) for d in selected_days]
        )
        cp_score_aligned = change_point_features.score.reindex(cp_session_index)
        cp_days_since_aligned = change_point_features.days_since_last_break.reindex(
            cp_session_index
        )
        cp_method = change_point_features.method
    else:
        cp_score_aligned = None
        cp_days_since_aligned = None
        cp_method = None

    clustering_features = feature_store.clustering
    if clustering_features is not None:
        session_index = pd.DatetimeIndex(
            [pd.Timestamp(d) for d in selected_days]
        )
        cluster_id_aligned = clustering_features.cluster_id.reindex(session_index)
        cluster_distance_aligned = clustering_features.distance_to_centroid.reindex(
            session_index
        )
        cluster_model_version = clustering_features.model_version
    else:
        cluster_id_aligned = None
        cluster_distance_aligned = None
        cluster_model_version = None
    trend_direction_outputs = axis_bundle.trend_direction.outputs_by_date
    trend_character_outputs = axis_bundle.trend_character.outputs_by_date
    volatility_outputs = axis_bundle.volatility_state.outputs_by_date
    breadth_outputs = axis_bundle.breadth_state.outputs_by_date
    event_outputs = axis_bundle.event_calendar
    network_fragility_by_date = _resolve_network_fragility_by_date(
        bundle_entry=axis_bundle.network_fragility,
        sessions=working_context.sessions,
    )
    # v2 §1E volume/liquidity axis (Slice 2.7). Stays None when the v2
    # config / volume seam is absent — preserves V1 byte-identity since
    # RegimeOutput.volume_liquidity_state already defaults to None.
    volume_liquidity_by_date = axis_bundle.volume_liquidity_state
    # v2 §2C credit/funding axis (Slice 4). Stays None when the v2 config /
    # cross_asset / macro seams are absent — preserves V1 byte-identity
    # since RegimeOutput.credit_funding_state already defaults to None.
    credit_funding_by_date = axis_bundle.credit_funding
    # v2 §2C credit/funding PROXY axis (Ambiguity Log #71) — the TLT-vs-HYG/LQD
    # differential run. Parallel to credit_funding; downstream consumers use
    # the explicit effective resolver.
    credit_funding_proxy_by_date = axis_bundle.credit_funding_proxy
    credit_funding_effective_by_date = axis_bundle.credit_funding_effective
    # v2 §2A monetary pressure axis (Ambiguity Log #46). Stays None when the
    # v2 config / macro_series seam is absent — preserves V1 byte-identity
    # since RegimeOutput.monetary_pressure_state defaults to None.
    monetary_pressure_state_by_date = axis_bundle.monetary_pressure_state
    # v2 §2B inflation/growth axis (Slice 5). Stays None when the v2 config /
    # macro_series / cross_asset seams are absent — preserves V1 byte-identity
    # since RegimeOutput.inflation_growth_state defaults to None.
    inflation_growth_by_date = axis_bundle.inflation_growth
    cohort_routing_config = working_context.config.cohort_routing

    outputs: list[RegimeOutput] = []
    for idx, day in enumerate(selected_days):
        trend_direction_output = trend_direction_outputs[day]
        trend_character_output = trend_character_outputs[day]
        volatility_output = volatility_outputs[day]
        _raw_breadth = breadth_outputs[day]
        assert isinstance(_raw_breadth, BreadthStateOutput), f"breadth_state produced {type(_raw_breadth).__name__}, expected BreadthStateOutput"
        breadth_output = _raw_breadth
        event_output = event_outputs[day]
        transition_output = transition_risk[day]
        network_fragility_output = network_fragility_by_date[day]
        volume_liquidity_output = (
            volume_liquidity_by_date.get(day)
            if volume_liquidity_by_date is not None
            else None
        )
        credit_funding_output = (
            credit_funding_by_date.get(day)
            if credit_funding_by_date is not None
            else None
        )
        credit_funding_proxy_output = (
            credit_funding_proxy_by_date.get(day)
            if credit_funding_proxy_by_date is not None
            else None
        )
        credit_funding_effective_output = (
            credit_funding_effective_by_date.get(day)
            if credit_funding_effective_by_date is not None
            else None
        )
        monetary_pressure_output = (
            monetary_pressure_state_by_date.get(day)
            if monetary_pressure_state_by_date is not None
            else None
        )
        monetary_pressure = (
            MonetaryPressureOutput(
                label=monetary_pressure_output.active_label,
                evidence=monetary_pressure_output.evidence,
                data_quality=monetary_pressure_output.data_quality,
            )
            if monetary_pressure_output is not None
            else MonetaryPressureOutput(
                label="unknown",
                evidence={"reason": "v2_classifier_not_yet_implemented"},
                data_quality=_v2_classifier_not_yet_implemented_data_quality(),
            )
        )
        inflation_growth_output = (
            inflation_growth_by_date.get(day)
            if inflation_growth_by_date is not None
            else None
        )
        change_point_output: ChangePointOutput | None = None
        if (
            cp_score_aligned is not None
            and cp_days_since_aligned is not None
            and cp_method is not None
        ):
            score_val = cp_score_aligned.iloc[idx]
            if score_val is not None and not pd.isna(score_val):
                days_val = cp_days_since_aligned.iloc[idx]
                days_since_int: int | None
                if days_val is None or pd.isna(days_val):
                    days_since_int = None
                else:
                    days_since_int = int(days_val)
                change_point_output = ChangePointOutput(
                    score=float(score_val),
                    days_since_last_break=days_since_int,
                    method=cp_method,
                )

        cluster_output: ClusterOutput | None = None
        if (
            cluster_id_aligned is not None
            and cluster_distance_aligned is not None
            and cluster_model_version is not None
        ):
            cid_val = cluster_id_aligned.iloc[idx]
            dist_val = cluster_distance_aligned.iloc[idx]
            if cid_val is not None and not pd.isna(cid_val) and not pd.isna(dist_val):
                cluster_output = ClusterOutput(
                    cluster_id=int(cid_val),
                    distance_to_centroid=float(dist_val),
                    model_version=cluster_model_version,
                )
        agent_routing = None
        strategy_family_constraints = None
        if cohort_routing_config is not None:
            # v2 §2A monetary pressure axis (Ambiguity Log #46) — wire the
            # active label through to cohort routing when the axis is lit.
            monetary_label: str | None = None
            if monetary_pressure_output is not None:
                monetary_label = monetary_pressure_output.active_label
            agent_routing = evaluate_cohort_routing(
                trend_direction_active=trend_direction_output.active_label,
                trend_character_active=trend_character_output.active_label,
                volatility_state_active=volatility_output.active_label,
                breadth_state_active=breadth_output.active_label,
                network_fragility_active=network_fragility_output.active_label,
                monetary_pressure_active=monetary_label,
                config=cohort_routing_config,
            )
            sfc_config = working_context.config.strategy_family_constraints
            if sfc_config is not None and agent_routing is not None:
                strategy_family_constraints = resolve_strategy_family_constraints(
                    active_cohort=agent_routing.active_cohort,
                    config=sfc_config,
                )
        outputs.append(
            RegimeOutput(
                engine_version=engine_version(),
                config_version=working_context.config.config_version,
                as_of_date=day,
                # V1 wire contract: output.market is the classified proxy
                # instrument; RegimeConfig.market is the broader universe.
                market="SPY",
                trend_direction=trend_direction_output,
                trend_character=trend_character_output,
                volatility_state=volatility_output,
                breadth_state=breadth_output,
                structural_causal_state=StructuralCausalState(
                    event_calendar=event_output,
                    monetary_pressure=monetary_pressure,
                ),
                network_fragility=network_fragility_output,
                transition_risk=transition_output,
                strategy_response=build_strategy_response(
                    trend_direction_active=trend_direction_output.active_label,
                    trend_character_active=trend_character_output.active_label,
                    volatility_state_active=volatility_output.active_label,
                    breadth_state_active=breadth_output.active_label,
                    transition_risk_label=transition_output.label,
                    event_calendar_active=event_output.active_label,
                ),
                volume_liquidity_state=volume_liquidity_output,
                credit_funding_state=credit_funding_output,
                credit_funding_state_proxy=credit_funding_proxy_output,
                credit_funding_effective_state=credit_funding_effective_output,
                inflation_growth_state=inflation_growth_output,
                monetary_pressure_state=monetary_pressure_output,
                cluster=cluster_output,
                change_point=change_point_output,
                agent_routing=agent_routing,
                strategy_family_constraints=strategy_family_constraints,
            )
        )

    return RegimeTimeline(
        engine_version=engine_version(),
        config_version=working_context.config.config_version,
        # V1 wire contract: output.market is the classified proxy instrument;
        # RegimeConfig.market remains the broader universe identifier.
        market="SPY",
        start_date=selected_days[0],
        end_date=selected_days[-1],
        trading_calendar=working_context.config.trading_calendar,
        outputs=outputs,
    )
