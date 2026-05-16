"""v2 §1E Volume / Liquidity rule engine + precedence (Slice 2.7).

Pure scalar rule layer over the v2 §1E features:
- ``volume_zscore_20d`` from ``FeatureStore.volume_liquidity_v2`` (slice 2.4).
- ``return_1d`` from V1 ``VolatilityFeatures.return_1d``
  (single source of truth — see Ambiguity Log #42).
- ``gap_frequency_percentile_252d`` + ``intraday_range_percentile_252d``
  from the §1C ``volatility_state_v2`` seam for ``liquidity_gap_behavior``
  (see Ambiguity Log #40 closure).

Spec references (docs/regime_engine_v2_spec.md):
    §1E lines 260-266  Labels
    §1E lines 268-286  Rules
    §1E lines 288-294  Risk Rank
    §1E line  282      Precedence implicit:
                       panic_volume > liquidity_gap_behavior > normal_volume > unknown

The three rules are evaluated in precedence order; the first match wins.
If none match (or any required input is NaN), the label falls through to
``unknown`` (data-quality gate, mirrors slice 1.3 / §3.3 convention).

Slice scope:
- Ships ``panic_volume`` (§1E lines 270-274), ``normal_volume`` (§1E
  line 282), and ``liquidity_gap_behavior`` (§1E lines 276-280) live.
- The 252d percentile of ``gap_frequency_20d`` was previously the
  missing input that forced ``liquidity_gap_behavior`` to short-circuit
  to False (Log #40). It now ships from
  ``regime_detection.volatility_state_v2.gap_frequency_percentile_252d``
  in the same commit that flipped this predicate. Log #40 is closed.

All numeric thresholds are config-driven via
``VolumeLiquidityRulesConfig``; this module is magic-number free per
CLAUDE.md Constants rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from regime_detection.config import VolumeLiquidityRulesConfig
from regime_detection.data_quality import assess_series_input_quality, quality_forces_unknown
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.models import DataQuality, VolumeLiquidityStateOutput

if TYPE_CHECKING:
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


# v2 §1E lines 260-266 labels (full Literal).
VolumeLiquidityLabel = Literal[
    "normal_volume",
    "panic_volume",
    "liquidity_gap_behavior",
    "unknown",
]


# v2 §1E lines 288-294 — risk rank, verbatim from spec. Drives per-label
# asymmetric hysteresis. Pinned constant (NOT a tunable).
VOLUME_LIQUIDITY_RISK_RANK: dict[str, int] = {
    "normal_volume": 0,
    "unknown": 1,
    "liquidity_gap_behavior": 2,
    "panic_volume": 3,
}


# Precedence implicit from §1E line 282 "otherwise" — higher-risk rules win
# first, with `unknown` reserved as the data-quality fallback at the bottom.
RULE_PRECEDENCE: tuple[VolumeLiquidityLabel, ...] = (
    "panic_volume",
    "liquidity_gap_behavior",
    "normal_volume",
)


@dataclass(frozen=True)
class VolumeLiquidityRuleInputs:
    """Per-day scalar inputs the §1E rules consume.

    All four fields are populated because ``liquidity_gap_behavior``
    consumes the two percentile fields from ``volatility_state_v2``.
    """

    # §1E line 272 / 256 — z-score of today's volume vs trailing 20d.
    volume_zscore_20d: float
    # §1E line 273 — today's SPY total return.
    return_1d: float
    # §1E line 278 — 252d percentile of `gap_frequency_20d`.
    gap_frequency_percentile_252d: float
    # §1E line 279 — 252d percentile of intraday range.
    intraday_range_percentile_252d: float


def _is_nan(value: float) -> bool:
    return bool(np.isnan(value))


def evaluate_panic_volume(
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> bool:
    """v2 §1E lines 270-274.

    ``volume_zscore_20d > panic_volume_zscore_threshold (2.0)``
    AND ``return_1d < panic_volume_return_threshold (-0.02)``.

    Strict inequalities verbatim per spec. Any NaN input falsifies the
    rule (mirrors slice 1.3 / slice 2.6 cold-start contract — a
    partially-warmed-up session cannot trigger a risk-up override).
    """
    if _is_nan(inputs.volume_zscore_20d) or _is_nan(inputs.return_1d):
        return False
    return bool(
        inputs.volume_zscore_20d > config.panic_volume_zscore_threshold
        and inputs.return_1d < config.panic_volume_return_threshold
    )


def evaluate_liquidity_gap_behavior(
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> bool:
    """v2 §1E lines 276-280 — `liquidity_gap_behavior` predicate.

    Spec text:
        gap_frequency_20d percentile_252d > 0.75
        AND intraday_range_percentile_252d > 0.75

    Both inequalities are strict per spec (`> 0.75`). NaN in either
    percentile input falsifies the rule (V1 §2.7 cold-start contract).
    The two thresholds are configurable via
    `VolumeLiquidityRulesConfig.liquidity_gap_*_percentile_threshold`
    so the V2 §9.1 walk-forward calibration may retune.

    Closes Log #40 — the 252d percentile of `gap_frequency_20d` now
    ships from `regime_detection.volatility_state_v2` (in the same
    commit that flipped this predicate from `return False`), so both
    rule inputs are available at evaluation time.
    """
    if _is_nan(inputs.gap_frequency_percentile_252d) or _is_nan(
        inputs.intraday_range_percentile_252d
    ):
        return False
    return bool(
        inputs.gap_frequency_percentile_252d
        > config.liquidity_gap_frequency_percentile_threshold
        and inputs.intraday_range_percentile_252d
        > config.liquidity_gap_intraday_range_percentile_threshold
    )


def evaluate_normal_volume(
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> bool:
    """v2 §1E line 282 — "otherwise".

    True iff NEITHER ``panic_volume`` NOR ``liquidity_gap_behavior``
    fires, AND the required inputs are not NaN (so we can actually
    assert the negation). A NaN input maps to False here and the
    precedence walker falls through to ``unknown``.
    """
    if _is_nan(inputs.volume_zscore_20d) or _is_nan(inputs.return_1d):
        return False
    if evaluate_panic_volume(inputs, config):
        return False
    if evaluate_liquidity_gap_behavior(inputs, config):
        return False
    return True


def evaluate_rules(
    *,
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> VolumeLiquidityLabel:
    """Walk the §1E precedence and return the first matching label.

    Falls through to ``unknown`` when no rule fires (cold-start /
    data-quality path). Precedence:
        panic_volume > liquidity_gap_behavior > normal_volume > unknown
    """
    for label in RULE_PRECEDENCE:
        if label == "panic_volume":
            if evaluate_panic_volume(inputs, config):
                return "panic_volume"
        elif label == "liquidity_gap_behavior":
            if evaluate_liquidity_gap_behavior(inputs, config):
                return "liquidity_gap_behavior"
        elif label == "normal_volume":
            if evaluate_normal_volume(inputs, config):
                return "normal_volume"
    return "unknown"


def build_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, VolumeLiquidityStateOutput] | None:
    """Free-function replacement for VolumeLiquidityStateSeriesClassifier.build()."""
    volume_features = feature_store.volume_liquidity_v2
    if volume_features is None:
        return None

    volume_liquidity_config = context.config.volume_liquidity_state
    if volume_liquidity_config is None:
        return None

    return_1d_series = feature_store.volatility.return_1d
    volume_zscore_series = volume_features.volume_zscore_20d

    volatility_v2 = feature_store.volatility_state_v2
    gap_freq_pct_series: pd.Series | None = None
    intraday_pct_series: pd.Series | None = None
    if volatility_v2 is not None:
        gap_freq_pct_series = volatility_v2.gap_frequency_percentile_252d
        intraday_pct_series = volatility_v2.intraday_range_percentile_252d

    required_inputs: list[pd.Series] = [
        volume_zscore_series,
        return_1d_series,
    ]
    required_trading_days = 20
    max_freshness_days = context.config.data_quality.max_freshness_days
    min_completeness = context.config.data_quality.min_completeness

    raw_labels: list[VolumeLiquidityLabel] = []
    per_day_data_quality: list[DataQuality] = []
    per_day_evidence: list[dict[str, object]] = []

    for day in context.sessions:
        dt = pd.Timestamp(day)

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
            per_day_evidence.append({"reason": day_quality.reason or "insufficient_data"})
            continue

        volume_zscore_20d = float(volume_zscore_series.loc[dt]) if dt in volume_zscore_series.index else float("nan")
        return_1d = float(return_1d_series.loc[dt]) if dt in return_1d_series.index else float("nan")
        gap_freq_pct = (
            float(gap_freq_pct_series.loc[dt])
            if gap_freq_pct_series is not None and dt in gap_freq_pct_series.index
            else float("nan")
        )
        intraday_pct = (
            float(intraday_pct_series.loc[dt])
            if intraday_pct_series is not None and dt in intraday_pct_series.index
            else float("nan")
        )

        inputs = VolumeLiquidityRuleInputs(
            volume_zscore_20d=volume_zscore_20d,
            return_1d=return_1d,
            gap_frequency_percentile_252d=gap_freq_pct,
            intraday_range_percentile_252d=intraday_pct,
        )
        label = evaluate_rules(
            inputs=inputs,
            config=volume_liquidity_config.rules,
        )
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append({
            "rule_evidence": {
                "volume_zscore_20d": float(f"{volume_zscore_20d:.8g}"),
                "return_1d": float(f"{return_1d:.8g}"),
                "gap_frequency_percentile_252d": float(f"{gap_freq_pct:.8g}"),
                "intraday_range_percentile_252d": float(f"{intraday_pct:.8g}"),
            },
        })

    stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=VOLUME_LIQUIDITY_RISK_RANK,
        deescalation_days_by_label=volume_liquidity_config.deescalation_days_by_label,
        default_deescalation_days=volume_liquidity_config.default_deescalation_days,
    )

    outputs: dict[date, VolumeLiquidityStateOutput] = {}
    for day, raw, stable, active, dq, evidence in zip(
        context.sessions,
        raw_labels,
        stable_labels,
        active_labels,
        per_day_data_quality,
        per_day_evidence,
        strict=True,
    ):
        outputs[day] = VolumeLiquidityStateOutput(
            raw_label=raw,
            stable_label=stable,
            active_label=active,
            evidence=evidence,
            data_quality=dq,
        )
    return outputs
