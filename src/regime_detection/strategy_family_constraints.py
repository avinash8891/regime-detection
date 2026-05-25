"""v2 §5.2 Strategy Family Constraints resolver.

Pins V2 spec §5.2 (docs/regime_engine_v2_spec.md). The
resolver implements the override-on-default inheritance contract: each
cohort declares only the field-level deltas vs ``default_neutral``; the
unspecified families and unspecified fields inherit the baseline.
"""

from __future__ import annotations

from regime_detection.config import (
    FamilyOverride,
    StrategyFamilyConstraintsConfig,
)
from regime_detection.models import StrategyFamilyConstraint

# v2 §5.2 (spec lines 3899-3921) — the five strategy families the engine
# constrains. One home (AGENTS rule B) so test files and downstream
# consumers import the same tuple rather than hardcoding strings.
STRATEGY_FAMILIES: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "breakout",
    "short_vol",
    "long_vol",
)


# Field set carried by FamilyOverride / StrategyFamilyConstraint.
# Declared once so the merger and the model stay in lockstep.
_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "allowed",
    "max_lookback_days",
    "max_holding_days",
    "max_position_pct",
    "min_adx",
    "require_breadth_confirmation",
    "require_volume_confirmation",
    "event_window_only",
    "reason",
)


def resolve_strategy_family_constraints(
    *,
    active_cohort: str,
    config: StrategyFamilyConstraintsConfig,
) -> dict[str, StrategyFamilyConstraint]:
    """Resolve override-on-default inheritance for ``active_cohort``.

    For each family present in ``config.default_neutral``:
      1. Validate the baseline carries ``allowed`` (v2 §5.2 contract).
      2. If ``active_cohort`` has an override for this family, merge field
         by field — override-non-None replaces the default; override-None
         inherits the default.
      3. Return the merged StrategyFamilyConstraint.

    Unknown ``active_cohort`` values fall through to the default_neutral
    baseline (no overrides found → empty override dict → all fields
    inherit). This mirrors §5.1's universal-fallback contract.
    """
    cohort_overrides = config.overrides.get(active_cohort, {})
    result: dict[str, StrategyFamilyConstraint] = {}
    for family, default in config.default_neutral.items():
        if default.allowed is None:
            raise ValueError(
                f"default_neutral.{family}.allowed must be set "
                "(V2 §5.2 baseline contract)"
            )
        override = cohort_overrides.get(family)
        result[family] = _merge(default=default, override=override)
    return result


def _merge(
    *,
    default: FamilyOverride,
    override: FamilyOverride | None,
) -> StrategyFamilyConstraint:
    """Field-by-field override-on-default merge."""
    values: dict[str, object] = {}
    for field in _CONSTRAINT_FIELDS:
        default_value = getattr(default, field)
        override_value = getattr(override, field) if override is not None else None
        values[field] = override_value if override_value is not None else default_value
    if (
        override is not None
        and default.allowed is False
        and override.allowed is True
        and override.reason is None
    ):
        values["reason"] = None
    return StrategyFamilyConstraint(**values)
