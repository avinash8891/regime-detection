"""v2 §1E Volume / Liquidity rule engine + precedence.

Pure scalar rule layer over the v2 §1E features:
- ``volume_zscore_20d`` from ``FeatureStore.volume_liquidity_v2``.
- ``return_1d`` from V1 ``VolatilityFeatures.return_1d``
  (single source of truth).
- ``gap_frequency_percentile_252d`` + ``intraday_range_percentile_252d``
  from the §1C ``volatility_state_v2`` seam for ``liquidity_gap_behavior``
  (feature placement decision).

Spec references (docs/regime_engine_v2_spec.md):
    §1E lines 399-405  Labels
    §1E lines 407-424  Rules
    §1E lines 426-433  Risk Rank
    §1E line  423      Precedence implicit ("otherwise"):
                       panic_volume > liquidity_gap_behavior > normal_volume > unknown

The three rules are evaluated in precedence order; the first match wins.
If none match (or any required input is NaN), the label falls through to
``unknown`` (data-quality gate, mirrors the §3.3 convention).

This module ships ``panic_volume`` (§1E lines 409-413), ``normal_volume``
(§1E lines 421-423), and ``liquidity_gap_behavior`` (§1E lines 415-419) live.
The 252d percentile of ``gap_frequency_20d`` comes from
``regime_detection.volatility_state_v2.gap_frequency_percentile_252d``.

All numeric thresholds are config-driven via
``VolumeLiquidityRulesConfig``; this module is magic-number free per
CLAUDE.md Constants rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from regime_detection.config import VolumeLiquidityRulesConfig


# v2 §1E lines 399-405 labels (full Literal).
VolumeLiquidityLabel = Literal[
    "normal_volume",
    "panic_volume",
    "liquidity_gap_behavior",
    "unknown",
]


# v2 §1E lines 426-433 — risk rank, verbatim from spec. Drives per-label
# asymmetric hysteresis. Pinned constant (NOT a tunable).
VOLUME_LIQUIDITY_RISK_RANK: dict[VolumeLiquidityLabel, int] = {
    "normal_volume": 0,
    "unknown": 1,
    "liquidity_gap_behavior": 2,
    "panic_volume": 3,
}


# Precedence implicit from §1E line 423 "otherwise" — higher-risk rules win
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

    # §1E line 411 / 395 — z-score of today's volume vs trailing 20d.
    volume_zscore_20d: float
    # §1E line 412 — today's SPY total return.
    return_1d: float
    # §1E line 417 — 252d percentile of `gap_frequency_20d`.
    gap_frequency_percentile_252d: float
    # §1E line 418 — 252d percentile of intraday range.
    intraday_range_percentile_252d: float


@dataclass(frozen=True)
class VolumeLiquidityRuleEvaluation:
    label: VolumeLiquidityLabel
    rule_path: str
    reason: str | None = None


def _is_nan(value: float) -> bool:
    return bool(np.isnan(value))


def _gap_history_unavailable(inputs: VolumeLiquidityRuleInputs) -> bool:
    return _is_nan(inputs.gap_frequency_percentile_252d) or _is_nan(
        inputs.intraday_range_percentile_252d
    )


def evaluate_panic_volume(
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> bool:
    """v2 §1E lines 409-413.

    ``volume_zscore_20d > panic_volume_zscore_threshold (2.0)``
    AND ``return_1d < panic_volume_return_threshold (-0.02)``.

    Strict inequalities verbatim per spec. Any NaN input falsifies the
    rule (NaN-falsifies cold-start contract — a
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
    """v2 §1E lines 415-419 — `liquidity_gap_behavior` predicate.

    Spec text:
        gap_frequency_20d percentile_252d > 0.75
        AND intraday_range_percentile_252d > 0.75

    Both inequalities are strict per spec (`> 0.75`). NaN in either
    percentile input falsifies the rule (V1 §2.7 cold-start contract).
    The two thresholds are configurable via
    `VolumeLiquidityRulesConfig.liquidity_gap_*_percentile_threshold`
    so the V2 §9.1 walk-forward calibration may retune.

    Implements the feature-placement decision: the 252d percentile of `gap_frequency_20d` now
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
    """v2 §1E lines 421-423 — "otherwise".

    True iff NEITHER ``panic_volume`` NOR ``liquidity_gap_behavior``
    fires, AND the required inputs are not NaN (so we can actually
    assert the negation). A NaN input maps to False here and the
    precedence walker falls through to ``unknown``.
    """
    if (
        _is_nan(inputs.volume_zscore_20d)
        or _is_nan(inputs.return_1d)
        or _gap_history_unavailable(inputs)
    ):
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
    return evaluate_rules_with_evidence(inputs=inputs, config=config).label


def evaluate_rules_with_evidence(
    *,
    inputs: VolumeLiquidityRuleInputs,
    config: VolumeLiquidityRulesConfig,
) -> VolumeLiquidityRuleEvaluation:
    """Walk §1E precedence and report why `unknown` was emitted."""
    for label in RULE_PRECEDENCE:
        if label == "panic_volume":
            if evaluate_panic_volume(inputs, config):
                return VolumeLiquidityRuleEvaluation(
                    label="panic_volume",
                    rule_path="standard",
                )
        elif label == "liquidity_gap_behavior":
            if evaluate_liquidity_gap_behavior(inputs, config):
                return VolumeLiquidityRuleEvaluation(
                    label="liquidity_gap_behavior",
                    rule_path="percentile",
                )
        elif label == "normal_volume":
            if evaluate_normal_volume(inputs, config):
                return VolumeLiquidityRuleEvaluation(
                    label="normal_volume",
                    rule_path="standard",
                )
    reason = "insufficient_gap_history" if _gap_history_unavailable(inputs) else "no_rule_fired"
    return VolumeLiquidityRuleEvaluation(
        label="unknown",
        rule_path="none",
        reason=reason,
    )
