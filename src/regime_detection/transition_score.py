"""Transition-pressure score composer.

Computes normalized component scores and returns a single 0..1 pressure score
with component evidence. Every configured component is required; missing inputs
raise instead of being reweighted away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from regime_detection.config import TransitionScoreConfig

EVENT_SET: frozenset[str] = frozenset(
    {
        "fed_week",
        "cpi_week",
        "nfp_week",
        "budget_week",
        "election_window",
        "global_rate_decision",
    }
)

_CREDIT_LABEL_SCORE: dict[str, float | None] = {
    "credit_calm": 0.0,
    "credit_recovery": 0.20,
    "credit_divergence": 0.25,
    "spread_widening": 0.45,
    "credit_stress": 0.75,
    "funding_squeeze": 0.90,
    "deleveraging": 1.0,
    "unknown": None,
}

_LIQUIDITY_LABEL_SCORE: dict[str, float | None] = {
    "normal_volume": 0.0,
    "liquidity_gap_behavior": 0.70,
    "panic_volume": 1.0,
    "unknown": None,
}


@dataclass(frozen=True)
class ComposedTransitionScore:
    score: float | None
    interpretation: str | None
    components: dict[str, float] | None
    missing_components: tuple[str, ...] = ()
    component_weight_coverage: float = 0.0
    macro_event_labels: tuple[str, ...] = ()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _optional_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def _max_present(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _label_score(label: str | None, scores: dict[str, float | None]) -> float | None:
    if label is None:
        return None
    if label not in scores:
        raise ValueError(f"unknown transition-risk label score input: {label!r}")
    return scores[label]


def _macro_event_labels(event_calendar_labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(label for label in event_calendar_labels if label in EVENT_SET)


def compute_transition_score(
    *,
    components: dict[str, float | None],
    weights: dict[str, float],
    minimum_component_weight_coverage: float,
) -> tuple[float | None, dict[str, float] | None, tuple[str, ...], float]:
    unknown_keys = sorted(set(components) - set(weights))
    if unknown_keys:
        raise ValueError(f"components missing configured weights: {unknown_keys}")

    missing = tuple(key for key in weights if components.get(key) is None)
    present_weight = sum(
        weight for key, weight in weights.items() if components.get(key) is not None
    )
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("transition score weights must sum to a positive value")
    coverage = present_weight / total_weight
    if missing:
        raise RuntimeError("transition score missing components: " + ", ".join(missing))
    if coverage < minimum_component_weight_coverage:
        return None, None, missing, coverage

    present_components = {
        key: float(value)
        for key, value in components.items()
        if value is not None and key in weights
    }
    score = sum(
        present_components[key] * weights[key] / present_weight
        for key in present_components
    )
    return _clip(score, 0.0, 1.0), present_components, missing, coverage


def interpret_transition_score(
    score: float, bands: dict[str, tuple[float, float]]
) -> str:
    if score < 0.0 or score > 1.0:
        raise ValueError(f"transition_score must be in [0.0, 1.0], got {score!r}")
    if score < bands["weakening"][0]:
        return "stable"
    if score < bands["transition_warning"][0]:
        return "weakening"
    if score < bands["high"][0]:
        return "transition_warning"
    return "high"


def compose_transition_score_for_session(
    *,
    realized_vol_short: float,
    realized_vol_long: float,
    pct_above_50dma: float,
    avg_pairwise_corr_percentile_504d: float,
    drawdown_252d: float,
    event_calendar_labels: tuple[str, ...],
    config: TransitionScoreConfig,
    spy_close: float | None = None,
    spy_sma_50: float | None = None,
    largest_eigenvalue_share_percentile_504d: float | None = None,
    effective_rank_percentile_504d: float | None = None,
    absorption_ratio_top3: float | None = None,
    credit_funding_label: str | None = None,
    volume_liquidity_label: str | None = None,
    volume_zscore_20d: float | None = None,
    gap_frequency_percentile_252d: float | None = None,
    intraday_range_percentile_252d: float | None = None,
    hmm_top_state_prob_now: float | None = None,
    hmm_top_state_prob_5d_ago: float | None = None,
    change_point_score: float | None = None,
    cluster_id_now: int | None = None,
    cluster_id_5d_ago: int | None = None,
) -> ComposedTransitionScore:
    scales = config.scales
    rv_short = _optional_number(realized_vol_short)
    rv_long = _optional_number(realized_vol_long)
    pct_above_50 = _optional_number(pct_above_50dma)
    corr_pct = _optional_number(avg_pairwise_corr_percentile_504d)
    dd_252 = _optional_number(drawdown_252d)

    vol_acc = None
    if rv_short is not None and rv_long not in {None, 0.0}:
        vol_acc = _clip(
            (rv_short / rv_long - 1.0) / scales.vol_acc_full_stress_ratio,
            0.0,
            1.0,
        )

    breadth_det = None
    if pct_above_50 is not None:
        breadth_det = _clip(
            (scales.breadth_zero_stress_pct - pct_above_50)
            / scales.breadth_full_stress_range,
            0.0,
            1.0,
        )

    trend_drawdown = (
        None
        if dd_252 is None
        else _clip(-dd_252 / scales.drawdown_full_stress, 0.0, 1.0)
    )
    ma_break = None
    close = _optional_number(spy_close)
    sma_50 = _optional_number(spy_sma_50)
    if close is not None and sma_50 not in {None, 0.0}:
        ma_break = _clip(
            (sma_50 - close) / sma_50 / scales.ma_break_full_stress, 0.0, 1.0
        )
    trend_break = _max_present(trend_drawdown, ma_break)

    largest_pct = _optional_number(largest_eigenvalue_share_percentile_504d)
    effective_rank_pct = _optional_number(effective_rank_percentile_504d)
    absorption = _optional_number(absorption_ratio_top3)
    effective_rank_stress = (
        None if effective_rank_pct is None else 1.0 - effective_rank_pct
    )
    absorption_stress = (
        None
        if absorption is None
        else _clip(
            (absorption - scales.absorption_floor) / scales.absorption_range, 0.0, 1.0
        )
    )
    correlation_fragility = _max_present(
        corr_pct, largest_pct, effective_rank_stress, absorption_stress
    )

    credit_stress = _label_score(credit_funding_label, _CREDIT_LABEL_SCORE)

    liquidity_label_stress = _label_score(
        volume_liquidity_label, _LIQUIDITY_LABEL_SCORE
    )
    volume_z = _optional_number(volume_zscore_20d)
    volume_stress = (
        None
        if volume_z is None
        else _clip(
            (volume_z - scales.volume_zscore_floor) / scales.volume_zscore_range,
            0.0,
            1.0,
        )
    )
    gap_stress = _optional_number(gap_frequency_percentile_252d)
    intraday_stress = _optional_number(intraday_range_percentile_252d)
    liquidity_stress = _max_present(
        liquidity_label_stress,
        volume_stress,
        gap_stress,
        intraday_stress,
    )

    macro_event_labels = _macro_event_labels(event_calendar_labels)
    macro_event = 1.0 if macro_event_labels else 0.0

    hmm_shift = None
    hmm_now = _optional_number(hmm_top_state_prob_now)
    hmm_5d = _optional_number(hmm_top_state_prob_5d_ago)
    if hmm_now is not None and hmm_5d is not None:
        hmm_shift = _clip(abs(hmm_now - hmm_5d), 0.0, 1.0)
    cp_score = _optional_number(change_point_score)
    if cp_score is not None:
        cp_score = _clip(cp_score, 0.0, 1.0)
    if (
        hmm_now is None
        or hmm_5d is None
        or cp_score is None
        or cluster_id_now is None
        or cluster_id_5d_ago is None
    ):
        model_instability = None
    else:
        cluster_flip = 1.0 if cluster_id_now != cluster_id_5d_ago else 0.0
        model_instability = _max_present(hmm_shift, cp_score, cluster_flip)

    raw_components: dict[str, float | None] = {
        "trend_break": trend_break,
        "volatility_acceleration": vol_acc,
        "breadth_deterioration": breadth_det,
        "correlation_fragility": correlation_fragility,
        "credit_stress": credit_stress,
        "liquidity_stress": liquidity_stress,
        "macro_event": macro_event,
        "model_instability": model_instability,
    }
    score, components, missing, coverage = compute_transition_score(
        components=raw_components,
        weights=config.weights,
        minimum_component_weight_coverage=config.minimum_component_weight_coverage,
    )
    # compute_transition_score raises if any configured component is missing.
    # This guard remains for defensive misconfigured coverage thresholds.
    if score is None or components is None:
        return ComposedTransitionScore(
            score=None,
            interpretation=None,
            components=None,
            missing_components=missing,
            component_weight_coverage=coverage,
            macro_event_labels=macro_event_labels,
        )
    return ComposedTransitionScore(
        score=score,
        interpretation=interpret_transition_score(score, config.bands),
        components=components,
        missing_components=missing,
        component_weight_coverage=coverage,
        macro_event_labels=macro_event_labels,
    )
