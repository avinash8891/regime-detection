from __future__ import annotations

import logging
from datetime import date
from typing import NamedTuple, cast

import pandas as pd

from regime_detection.axis_series import AxisSeriesBundle, build_axis_series_bundle
from regime_detection.cohort_routing import evaluate_cohort_routing
from regime_detection.config import RegimeConfig
from regime_detection.feature_store import FeatureStore, build_feature_store
from regime_detection.market_context import (
    MarketContext,
    slice_context_to_recent_sessions,
)
from regime_detection.models import (
    AxisEvidencePayload,
    AxisOutput,
    BreadthStateOutput,
    ChangePointOutput,
    ClusterOutput,
    DataQuality,
    HmmOutput,
    NetworkFragilityEvidencePayload,
    NetworkFragilityOutput,
    RegimeOutput,
    RegimeTimeline,
    StructuralCausalState,
    TransitionRiskOutput,
)
from regime_detection.strategy_family_constraints import (
    resolve_strategy_family_constraints,
)
from regime_detection.strategy_constraints import resolve_effective_strategy_constraints
from regime_detection.strategy_response import build_strategy_response
from regime_detection.transition_risk_series import build_transition_risk_series
from regime_detection.versioning import engine_version

_LOGGER = logging.getLogger(__name__)

ENGINE_MINIMUM_HISTORY = 320


class _AlignedV2Evidence(NamedTuple):
    cp_score_aligned: pd.Series | None
    cp_days_since_aligned: pd.Series | None
    cp_method: str | None
    cluster_id_aligned: pd.Series | None
    cluster_distance_aligned: pd.Series | None
    cluster_model_version: str | None
    cluster_n_clusters: int | None
    hmm_top_state_aligned: pd.Series | None
    hmm_top_state_prob_aligned: pd.Series | None
    hmm_top_state_full: pd.Series | None
    hmm_n_states: int | None
    hmm_model_version: str | None


def _v2_classifier_seam_absent_data_quality() -> DataQuality:
    """DataQuality emitted when the V2 axis classifier returned no bundle
    entry for this session (seam absent), not when the classifier itself
    is missing — every V2 axis builder is shipped and produces real
    classifications when its inputs are present. The historical reason
    string ``v2_classifier_not_yet_implemented`` is preserved in the JSON
    contract for back-compat (asserted in tests/test_schema_and_timeline.py).

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

    Prefer the AxisSeriesBundle entry when present — the network_fragility
    classifier is fully implemented and supplies real classifications on
    every session for which sector ETF inputs were passed. Fall back to a
    v2 'unknown' placeholder per session only when the bundle entry is
    None (sector ETF data missing).
    """
    if bundle_entry is not None:
        return bundle_entry
    placeholder_dq = _v2_classifier_seam_absent_data_quality()
    return {
        day: NetworkFragilityOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence=NetworkFragilityEvidencePayload(
                reason="v2_classifier_not_yet_implemented"
            ),
            data_quality=placeholder_dq,
        )
        for day in sessions
    }


def _resolve_timeline_required_sessions(
    *,
    available_sessions: int,
    lookback_days: int,
    config: RegimeConfig,
) -> int:
    # Trainable v2 evidence layers (HMM, GMM clustering, BOCPD
    # change-point) each need the trailing ``training_window_days`` rows of
    # their inputs to fit. Extend the engine's minimum slicing window to
    # the LARGEST configured training window. Without this, disabling one
    # seam (e.g. change_point) but keeping another (e.g. HMM) would slice
    # the context down below HMM's training_window_days and the HMM seam
    # would silently return None for insufficient history.
    # Preserves V1's slicing window for callers that omit all three v2 configs
    # (change_point, hmm, clustering): max() collapses to ENGINE_MINIMUM_HISTORY,
    # matching what the V1 engine path uses.
    v2_min_history = ENGINE_MINIMUM_HISTORY
    trailing_component_lookback = 0
    if config.change_point is not None:
        # +21 absorbs the realized_vol_21d warmup so BOCPD sees a full
        # non-NaN training window on the trailing slice.
        v2_min_history = max(
            v2_min_history, config.change_point.training_window_days + 21
        )
    if config.hmm is not None:
        # +63 absorbs the deepest HMM input warmup (drawdown_63d /
        # avg_pairwise_corr_63d) so the trailing slice gives the GaussianHMM
        # fit a full non-NaN training window.
        v2_min_history = max(v2_min_history, config.hmm.training_window_days + 63)
        # transition_score.model_instability needs top_state_prob[t-5].
        # Keep five extra warmed sessions ahead of the emitted window so the
        # first requested output can use the same PIT evidence as later rows.
        trailing_component_lookback = max(trailing_component_lookback, 5)
    if config.clustering is not None:
        # +63 absorbs the deepest GMM input warmup (return_63d /
        # drawdown_63d / avg_pairwise_corr_63d) so the trailing slice gives
        # the GaussianMixture fit a full non-NaN training window.
        v2_min_history = max(
            v2_min_history, config.clustering.training_window_days + 63
        )
    if config.monetary_pressure_v2 is not None:
        mp_config = config.monetary_pressure_v2
        monetary_change_window = max(
            mp_config.yield_change_lookback_days,
            mp_config.rate_shock_lookback_days,
            mp_config.broad_usd_lookback_days,
        )
        # Monetary-pressure z-scores use a rolling normalizer over N-session
        # changes. Keep enough pre-window sessions so the first emitted day can
        # classify from real z-scores instead of cold-start NaNs.
        v2_min_history = max(
            v2_min_history,
            mp_config.zscore_normalizer_window_days + monetary_change_window,
        )
    return min(
        available_sessions,
        v2_min_history + lookback_days - 1 + trailing_component_lookback,
    )


def _align_v2_evidence_for_selected_days(
    *,
    feature_store: FeatureStore,
    selected_days: list[date],
    hmm_model_version: str | None = None,
) -> _AlignedV2Evidence:
    # v2 §6.2 GMM clustering evidence — bulk-reindex BEFORE the per-day
    # loop (matches the `_build_transition_score_inputs_by_date` pattern).
    # Per-day `.get(pd.Timestamp(day))` would re-scan the index
    # n_sessions times; one reindex + positional access keeps the loop O(N).
    # v2 §6.3 BOCPD change-point evidence — bulk-reindex BEFORE the
    # per-day loop, same pattern as clustering. Stays None when the seam is
    # absent (config off, or SPY history shorter than training_window_days).
    session_index = pd.DatetimeIndex([pd.Timestamp(d) for d in selected_days])
    change_point_features = feature_store.change_point
    if change_point_features is not None:
        cp_score_aligned = change_point_features.score.reindex(session_index)
        cp_days_since_aligned = change_point_features.days_since_last_break.reindex(
            session_index
        )
        cp_method = change_point_features.method
    else:
        cp_score_aligned = None
        cp_days_since_aligned = None
        cp_method = None

    clustering_features = feature_store.clustering
    if clustering_features is not None:
        cluster_id_aligned = clustering_features.cluster_id.reindex(session_index)
        cluster_distance_aligned = clustering_features.distance_to_centroid.reindex(
            session_index
        )
        cluster_model_version = clustering_features.model_version
        cluster_n_clusters = clustering_features.n_clusters
    else:
        cluster_id_aligned = None
        cluster_distance_aligned = None
        cluster_model_version = None
        cluster_n_clusters = None

    hmm_features = feature_store.hmm
    if hmm_features is not None:
        proba = hmm_features.state_probabilities
        valid_rows = ~proba.isna().all(axis=1)
        hmm_top_state_full = pd.Series(pd.NA, index=proba.index, dtype="Int64")
        if valid_rows.any():
            hmm_top_state_full.loc[valid_rows] = (
                proba.loc[valid_rows].idxmax(axis=1).astype("Int64")
            )
        hmm_top_state_aligned = hmm_top_state_full.reindex(session_index)
        hmm_top_state_prob_aligned = hmm_features.top_state_prob.reindex(session_index)
        hmm_n_states = hmm_features.n_states
        hmm_model_version = hmm_model_version
    else:
        hmm_top_state_full = None
        hmm_top_state_aligned = None
        hmm_top_state_prob_aligned = None
        hmm_n_states = None
        hmm_model_version = None

    return _AlignedV2Evidence(
        cp_score_aligned=cp_score_aligned,
        cp_days_since_aligned=cp_days_since_aligned,
        cp_method=cp_method,
        cluster_id_aligned=cluster_id_aligned,
        cluster_distance_aligned=cluster_distance_aligned,
        cluster_model_version=cluster_model_version,
        cluster_n_clusters=cluster_n_clusters,
        hmm_top_state_aligned=hmm_top_state_aligned,
        hmm_top_state_prob_aligned=hmm_top_state_prob_aligned,
        hmm_top_state_full=hmm_top_state_full,
        hmm_n_states=hmm_n_states,
        hmm_model_version=hmm_model_version,
    )


def _hmm_state_persistence_days(
    full_top_state_series: pd.Series,
    target_timestamp: pd.Timestamp,
) -> int | None:
    """Count consecutive sessions the top state has been unchanged up to target_timestamp.

    Uses the full (un-windowed) HMM top-state series so persistence
    counts that started before the output window are reported correctly.
    """
    if target_timestamp not in full_top_state_series.index:
        return None
    loc = full_top_state_series.index.get_loc(target_timestamp)
    if not isinstance(loc, int):
        loc = int(loc)
    current_state = full_top_state_series.iloc[loc]
    if pd.isna(current_state):
        return None
    days = 1
    for i in range(loc - 1, -1, -1):
        prev = full_top_state_series.iloc[i]
        if pd.isna(prev) or int(prev) != int(current_state):
            break
        days += 1
    return days


def _enrich_with_hmm_evidence(
    output: AxisOutput,
    aligned: _AlignedV2Evidence,
    day_index: int,
) -> AxisOutput:
    if (
        aligned.hmm_top_state_prob_aligned is None
        or aligned.hmm_top_state_aligned is None
    ):
        return output
    prob = aligned.hmm_top_state_prob_aligned.iloc[day_index]
    state = aligned.hmm_top_state_aligned.iloc[day_index]
    if pd.isna(prob) or pd.isna(state):
        return output
    enriched = dict(output.evidence.root)
    enriched["hmm_top_state"] = int(state)
    enriched["hmm_top_state_prob"] = float(prob)
    return output.model_copy(update={"evidence": AxisEvidencePayload(root=enriched)})


def _build_change_point_output(
    *,
    aligned: _AlignedV2Evidence,
    selected_day_index: int,
) -> ChangePointOutput | None:
    if (
        aligned.cp_score_aligned is None
        or aligned.cp_days_since_aligned is None
        or aligned.cp_method is None
    ):
        return None
    score_val = aligned.cp_score_aligned.iloc[selected_day_index]
    if score_val is None or pd.isna(score_val):
        return None
    days_val = aligned.cp_days_since_aligned.iloc[selected_day_index]
    days_since_int = None if days_val is None or pd.isna(days_val) else int(days_val)
    return ChangePointOutput(
        score=float(score_val),
        days_since_last_break=days_since_int,
        method=aligned.cp_method,
    )


def _build_cluster_output(
    *,
    aligned: _AlignedV2Evidence,
    working_context: MarketContext,
    selected_day_index: int,
) -> ClusterOutput | None:
    if (
        aligned.cluster_id_aligned is None
        or aligned.cluster_distance_aligned is None
        or aligned.cluster_model_version is None
    ):
        return None
    cid_val = aligned.cluster_id_aligned.iloc[selected_day_index]
    dist_val = aligned.cluster_distance_aligned.iloc[selected_day_index]
    if cid_val is None or pd.isna(cid_val) or pd.isna(dist_val):
        return None

    clustering_config = working_context.config.clustering
    cluster_label_map = (
        clustering_config.cluster_label_map if clustering_config is not None else None
    )
    validated_cluster_label: str | None = None
    cluster_mapping_status = "map_absent"
    cluster_mapping_reason = "cluster_label_map_not_configured"
    if clustering_config is not None and cluster_label_map is None:
        if clustering_config.label_map_required_for_output:
            cluster_mapping_status = "map_required_missing"
            cluster_mapping_reason = "cluster_label_map_required_but_not_configured"
    elif cluster_label_map is not None and clustering_config is not None:
        map_covers_clusters = set(cluster_label_map.keys()) == set(
            range(clustering_config.n_clusters)
        )
        version_matches = (
            aligned.cluster_model_version == clustering_config.model_version
        )
        if map_covers_clusters and version_matches:
            validated_cluster_label = cluster_label_map.get(int(cid_val))
            cluster_mapping_status = "mapped"
            cluster_mapping_reason = "cluster_label_map_valid"
        elif not map_covers_clusters:
            cluster_mapping_status = "map_invalid"
            cluster_mapping_reason = "cluster_label_map_incomplete"
        else:
            cluster_mapping_status = "model_version_mismatch"
            cluster_mapping_reason = "cluster_label_map_model_version_mismatch"
        if not map_covers_clusters or not version_matches:
            if selected_day_index == 0:
                _LOGGER.warning(
                    "cluster_label_map skipped: map keys %s do not cover "
                    "n_clusters=%d or model_version mismatch (map=%s, fit=%s)",
                    sorted(cluster_label_map.keys()),
                    clustering_config.n_clusters,
                    clustering_config.model_version,
                    aligned.cluster_model_version,
                )
    return ClusterOutput(
        cluster_id=int(cid_val),
        distance_to_centroid=float(dist_val),
        model_version=aligned.cluster_model_version,
        mapped_label=validated_cluster_label,
        mapping_status=cluster_mapping_status,
        mapping_reason=cluster_mapping_reason,
    )


def _build_hmm_output(
    *,
    aligned: _AlignedV2Evidence,
    working_context: MarketContext,
    selected_day_index: int,
    day: date,
) -> HmmOutput | None:
    if (
        aligned.hmm_top_state_aligned is None
        or aligned.hmm_top_state_prob_aligned is None
        or aligned.hmm_n_states is None
    ):
        return None
    hmm_state_val = aligned.hmm_top_state_aligned.iloc[selected_day_index]
    hmm_prob_val = aligned.hmm_top_state_prob_aligned.iloc[selected_day_index]
    if hmm_state_val is None or pd.isna(hmm_state_val) or pd.isna(hmm_prob_val):
        return None

    persistence = (
        _hmm_state_persistence_days(aligned.hmm_top_state_full, pd.Timestamp(day))
        if aligned.hmm_top_state_full is not None
        else None
    )
    hmm_config = working_context.config.hmm
    hmm_label_map = hmm_config.state_label_map if hmm_config is not None else None
    validated_hmm_label: str | None = None
    hmm_mapping_status = "map_absent"
    hmm_mapping_reason = "state_label_map_not_configured"
    if hmm_config is not None and hmm_label_map is None:
        if hmm_config.label_map_required_for_output:
            hmm_mapping_status = "map_required_missing"
            hmm_mapping_reason = "state_label_map_required_but_not_configured"
    elif hmm_label_map is not None and hmm_config is not None:
        map_covers_states = set(hmm_label_map.keys()) == set(range(hmm_config.n_states))
        version_matches = (
            aligned.hmm_model_version or "hmm_unknown"
        ) == hmm_config.model_version
        if map_covers_states and version_matches:
            validated_hmm_label = hmm_label_map.get(int(hmm_state_val))
            hmm_mapping_status = "mapped"
            hmm_mapping_reason = "state_label_map_valid"
        elif not map_covers_states:
            hmm_mapping_status = "map_invalid"
            hmm_mapping_reason = "state_label_map_incomplete"
        else:
            hmm_mapping_status = "model_version_mismatch"
            hmm_mapping_reason = "state_label_map_model_version_mismatch"
        if not map_covers_states or not version_matches:
            if selected_day_index == 0:
                _LOGGER.warning(
                    "state_label_map skipped: map keys %s do not cover "
                    "n_states=%d or model_version mismatch (map=%s, fit=%s)",
                    sorted(hmm_label_map.keys()),
                    hmm_config.n_states,
                    hmm_config.model_version,
                    aligned.hmm_model_version,
                )
    return HmmOutput(
        top_state=int(hmm_state_val),
        top_state_prob=float(hmm_prob_val),
        n_states=aligned.hmm_n_states,
        state_persistence_days=persistence,
        model_version=aligned.hmm_model_version or "hmm_unknown",
        mapped_label=validated_hmm_label,
        mapping_status=hmm_mapping_status,
        mapping_reason=hmm_mapping_reason,
    )


def _build_timeline_output_for_day(
    *,
    day: date,
    selected_day_index: int,
    working_context: MarketContext,
    axis_bundle: AxisSeriesBundle,
    transition_risk: dict[date, TransitionRiskOutput],
    network_fragility_by_date: dict[date, NetworkFragilityOutput],
    aligned_v2_evidence: _AlignedV2Evidence,
) -> RegimeOutput:
    trend_direction_output = _enrich_with_hmm_evidence(
        axis_bundle.trend_direction.outputs_by_date[day],
        aligned_v2_evidence,
        selected_day_index,
    )
    trend_character_output = axis_bundle.trend_character.outputs_by_date[day]
    volatility_output = _enrich_with_hmm_evidence(
        axis_bundle.volatility_state.outputs_by_date[day],
        aligned_v2_evidence,
        selected_day_index,
    )
    breadth_output = cast(
        BreadthStateOutput, axis_bundle.breadth_state.outputs_by_date[day]
    )
    event_output = axis_bundle.event_calendar[day]
    transition_output = transition_risk[day]
    network_fragility_output = network_fragility_by_date[day]
    volume_liquidity_output = (
        axis_bundle.volume_liquidity_state.get(day)
        if axis_bundle.volume_liquidity_state is not None
        else None
    )
    credit_funding_output = (
        axis_bundle.credit_funding.get(day)
        if axis_bundle.credit_funding is not None
        else None
    )
    credit_funding_proxy_output = (
        axis_bundle.credit_funding_proxy.get(day)
        if axis_bundle.credit_funding_proxy is not None
        else None
    )
    credit_funding_effective_output = (
        axis_bundle.credit_funding_effective.get(day)
        if axis_bundle.credit_funding_effective is not None
        else None
    )
    monetary_pressure_output = (
        axis_bundle.monetary_pressure_state.get(day)
        if axis_bundle.monetary_pressure_state is not None
        else None
    )
    if (
        working_context.config.monetary_pressure_state is not None
        and monetary_pressure_output is None
    ):
        raise RuntimeError(
            "monetary_pressure_state is configured but unavailable; "
            "supply macro_series keys 2y_yield, 10y_yield, and broad_usd_index"
        )
    inflation_growth_output = (
        axis_bundle.inflation_growth.get(day)
        if axis_bundle.inflation_growth is not None
        else None
    )
    change_point_output = _build_change_point_output(
        aligned=aligned_v2_evidence,
        selected_day_index=selected_day_index,
    )
    cluster_output = _build_cluster_output(
        aligned=aligned_v2_evidence,
        working_context=working_context,
        selected_day_index=selected_day_index,
    )
    hmm_output = _build_hmm_output(
        aligned=aligned_v2_evidence,
        working_context=working_context,
        selected_day_index=selected_day_index,
        day=day,
    )

    agent_routing = None
    strategy_family_constraints = None
    effective_strategy_constraints = None
    cohort_routing_config = working_context.config.cohort_routing
    if cohort_routing_config is not None:
        # v2 §2A monetary pressure axis — wire the active label through to
        # cohort routing when the axis is lit (see spec §2A in
        # docs/regime_engine_v2_spec.md).
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
    strategy_response = build_strategy_response(
        trend_direction_active=trend_direction_output.active_label,
        trend_character_active=trend_character_output.active_label,
        volatility_state_active=volatility_output.active_label,
        breadth_state_active=breadth_output.active_label,
        transition_risk_state=transition_output.state,
        event_calendar_labels=event_output.matching_labels,
        event_modifier_config=working_context.config.strategy_event_modifiers,
    )
    if agent_routing is not None or strategy_family_constraints is not None:
        effective_strategy_constraints = resolve_effective_strategy_constraints(
            strategy_response=strategy_response,
            agent_routing=agent_routing,
            strategy_family_constraints=strategy_family_constraints,
        )
    return RegimeOutput(
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
        ),
        network_fragility=network_fragility_output,
        transition_risk=transition_output,
        strategy_response=strategy_response,
        volume_liquidity_state=volume_liquidity_output,
        credit_funding_state=credit_funding_output,
        credit_funding_state_proxy=credit_funding_proxy_output,
        credit_funding_effective_state=credit_funding_effective_output,
        inflation_growth_state=inflation_growth_output,
        monetary_pressure_state=monetary_pressure_output,
        hmm=hmm_output,
        cluster=cluster_output,
        change_point=change_point_output,
        agent_routing=agent_routing,
        strategy_family_constraints=strategy_family_constraints,
        effective_strategy_constraints=effective_strategy_constraints,
    )


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

    required_sessions = _resolve_timeline_required_sessions(
        available_sessions=len(context.sessions),
        lookback_days=lookback_days,
        config=cfg,
    )
    working_context = slice_context_to_recent_sessions(
        context=context,
        required_sessions=required_sessions,
    )
    is_v2 = cfg.config_version != "core3-v1.0.0"
    network_fragility_config = cfg.network_fragility if is_v2 else None
    trend_direction_v2_config = cfg.trend_direction_v2 if is_v2 else None
    volatility_state_v2_config = cfg.volatility_state_v2 if is_v2 else None
    breadth_state_v2_config = cfg.breadth_state_v2 if is_v2 else None
    volume_liquidity_v2_config = cfg.volume_liquidity_v2 if is_v2 else None
    monetary_pressure_v2_config = cfg.monetary_pressure_v2 if is_v2 else None
    credit_funding_config = cfg.credit_funding if is_v2 else None
    inflation_growth_config = cfg.inflation_growth if is_v2 else None
    central_bank_text_config = cfg.central_bank_text if is_v2 else None
    news_sentiment_config = cfg.news_sentiment if is_v2 else None
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
    axis_bundle = build_axis_series_bundle(
        context=working_context, feature_store=feature_store
    )
    selected_days = list(working_context.sessions[-lookback_days:])
    transition_risk = build_transition_risk_series(
        context=working_context,
        feature_store=feature_store,
        axis_bundle=axis_bundle,
        output_sessions=selected_days,
    )

    hmm_config = working_context.config.hmm
    aligned_v2_evidence = _align_v2_evidence_for_selected_days(
        feature_store=feature_store,
        selected_days=selected_days,
        hmm_model_version=hmm_config.model_version if hmm_config is not None else None,
    )
    network_fragility_by_date = _resolve_network_fragility_by_date(
        bundle_entry=axis_bundle.network_fragility,
        sessions=working_context.sessions,
    )
    outputs: list[RegimeOutput] = []
    for idx, day in enumerate(selected_days):
        outputs.append(
            _build_timeline_output_for_day(
                day=day,
                selected_day_index=idx,
                working_context=working_context,
                axis_bundle=axis_bundle,
                transition_risk=transition_risk,
                network_fragility_by_date=network_fragility_by_date,
                aligned_v2_evidence=aligned_v2_evidence,
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
