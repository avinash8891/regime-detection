"""v2 §1C Volatility axis — classify layer.

V1 raw-label walker (``raw_label_for_day`` / ``build_raw_outputs``) plus the
v2 §1C precedence overlay (``evaluate_rising_vol`` / ``evaluate_vol_crush``
/ ``evaluate_v2_volatility_label``). The features layer lives in
``volatility_state.py``.

Precedence (v2 §1C line 311):
    crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
    normal_vol > unknown
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.config import VolatilityV2RulesConfig
from regime_detection.volatility_state import (
    VolatilityFeatures,
    VolatilityV2Features,
)
from regime_shared.pandas_compat import require_single_session

# v2 §1C line 311 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
# `rising_vol` added per v2 §1C lines 250-254;
# `vol_crush` added per ADR 0005 using FRED VIXCLS as
# implied_vol_30d plus the event-window seam.
VolatilityLabel = Literal[
    "low_vol",
    "normal_vol",
    "high_vol",
    "crisis_vol",
    "unknown",
    "rising_vol",
    "vol_crush",
]


# v2 §1C line 311 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
#   normal_vol > unknown.
# V1 risk-rank contract is frozen in replay fixtures; crisis_vol remains 3.
# V2 crisis-vs-vol_crush precedence is resolved before hysteresis in
# this module's evaluate_v2_volatility_label, not by changing the V1
# evidence rank.
_RISK_RANK: dict[VolatilityLabel, int] = {
    "low_vol": 0,
    "normal_vol": 1,
    "high_vol": 2,
    "crisis_vol": 3,
    "unknown": 2,
    "rising_vol": 2,
    "vol_crush": 3,
}


def raw_label_for_day(
    f: VolatilityFeatures,
    dt: pd.Timestamp,
    *,
    volatility_state_v2_features: VolatilityV2Features | None = None,
    volatility_state_v2_rules: VolatilityV2RulesConfig | None = None,
) -> tuple[VolatilityLabel, dict[str, Any]]:
    """Per-day raw volatility_state label.

    When ``volatility_state_v2_features`` AND ``volatility_state_v2_rules``
    are both supplied, the v2 §1C precedence (line 191:
    ``crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
    normal_vol > unknown``) is layered ON TOP of the v1 label. When either
    is ``None`` the function returns the v1 label and evidence unchanged.

    F-043: this is a thin wrapper over :func:`build_raw_outputs` so the §5.5
    rule predicates, evidence shape, and v2 override have a single encoding.
    Slicing each feature to ``[dt]`` is safe because the vectorized builder and
    ``evaluate_v2_volatility_label`` only read values at the target session.
    """
    # Guard: dt must resolve to exactly one session — a duplicate-date index would make
    # .loc[[dt]] return multiple rows and labels[0] silently mask the data issue.
    require_single_session(f.close.index, dt)
    day_features = VolatilityFeatures(
        close=f.close.loc[[dt]],
        return_1d=f.return_1d.loc[[dt]],
        return_5d=f.return_5d.loc[[dt]],
        return_21d=f.return_21d.loc[[dt]],
        realized_vol_21d=f.realized_vol_21d.loc[[dt]],
        realized_vol_percentile_252d=f.realized_vol_percentile_252d.loc[[dt]],
        vix_percentile_252d=(
            None if f.vix_percentile_252d is None else f.vix_percentile_252d.loc[[dt]]
        ),
    )
    labels, evidence = build_raw_outputs(
        day_features,
        volatility_state_v2_features=volatility_state_v2_features,
        volatility_state_v2_rules=volatility_state_v2_rules,
    )
    return labels[0], evidence[0]


def build_raw_outputs(
    f: VolatilityFeatures,
    *,
    volatility_state_v2_features: VolatilityV2Features | None = None,
    volatility_state_v2_rules: VolatilityV2RulesConfig | None = None,
) -> tuple[list[VolatilityLabel], list[dict[str, Any]]]:
    """Vectorized v1 raw labels + optional v2 §1C `rising_vol` override.

    When both v2 args are supplied, the v2 §1C precedence at line 191 is applied per-day AFTER
    the v1 pass — `rising_vol` overrides v1 `low_vol` / `normal_vol` /
    `unknown` (NOT `crisis_vol` / `high_vol`, which outrank `rising_vol`).
    When either argument is None, output is byte-identical to v1.
    """
    ret1 = f.return_1d
    ret5 = f.return_5d
    ret21 = f.return_21d
    vol_pct = f.realized_vol_percentile_252d
    valid = ~(ret1.isna() | ret5.isna() | ret21.isna() | vol_pct.isna())

    rv21 = f.realized_vol_21d
    vix_pct_series = pd.Series(float("nan"), index=ret1.index, dtype=float)
    vix_present = pd.Series(False, index=ret1.index, dtype="bool")
    vix_crisis = pd.Series(False, index=ret1.index, dtype="bool")
    vix_high = pd.Series(False, index=ret1.index, dtype="bool")
    if f.vix_percentile_252d is not None:
        vix_pct_series = f.vix_percentile_252d.reindex(ret1.index)
        vix_present = vix_pct_series.notna()
        vix_crisis = vix_present & vix_pct_series.ge(0.95)
        vix_high = vix_present & vix_pct_series.ge(0.80)

    crisis = valid & (
        ret1.le(-0.05)
        | ret5.le(-0.08)
        | (vol_pct.ge(0.90) & ret21.le(-0.05))
        | vix_crisis
    )
    high_vol = valid & (vol_pct.ge(0.80) | vix_high)
    low_vol = valid & vol_pct.le(0.30)
    normal_vol = valid & ~(crisis | high_vol | low_vol)

    labels = np.full(len(ret1), "unknown", dtype=object)
    labels[normal_vol.to_numpy()] = "normal_vol"
    labels[low_vol.to_numpy()] = "low_vol"
    labels[high_vol.to_numpy()] = "high_vol"
    labels[crisis.to_numpy()] = "crisis_vol"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "realized_vol_21d": _ev_float(rv21.iat[idx]),
                "realized_vol_percentile_252d": _ev_float(vol_pct.iat[idx]),
                "vix_percentile_252d": (
                    _ev_float(vix_pct_series.iat[idx])
                    if bool(vix_present.iat[idx])
                    else None
                ),
                "crisis_vol": bool(crisis.iat[idx]),
                "high_vol": bool(high_vol.iat[idx]),
                "low_vol": bool(low_vol.iat[idx]),
            }
        )

    # Both v2 evidence enrichment (iv_rv_spread) and the v2 §1C label override apply
    # ONLY when BOTH v2 args are present — otherwise the output is byte-identical to v1
    # (docstring contract). Gating the iv_rv_spread block on features alone would change
    # the evidence shape on a partial v2 arg.
    if (
        volatility_state_v2_features is not None
        and volatility_state_v2_rules is not None
    ):
        iv_rv = volatility_state_v2_features.iv_rv_spread
        for idx, dt in enumerate(ret1.index):
            if labels[idx] != "unknown" and iv_rv is not None and dt in iv_rv.index:
                val = iv_rv.loc[dt]
                if not pd.isna(val):
                    evidence[idx]["iv_rv_spread"] = _ev_float(val)

        # v2 §1C line 311 precedence — applied per-day on top of v1.
        for idx, dt in enumerate(ret1.index):
            v1_label = str(labels[idx])
            v2_label = evaluate_v2_volatility_label(
                v1_label=v1_label,
                features=volatility_state_v2_features,
                dt=dt,
                rules_config=volatility_state_v2_rules,
            )
            if v2_label is None:
                continue
            evidence[idx]["v2_override"] = {
                "from": v1_label,
                "to": v2_label,
                "rule": v2_label,  # the winning v2 §1C rule (rising_vol / vol_crush)
            }
            labels[idx] = v2_label

    return list(labels), evidence


# ---------------------------------------------------------------------------
# v2 §1C `rising_vol` rule + precedence wrapper.
#
# Rule (v2 §1C lines 250-254, verbatim):
#     ATR_ratio > 1.15
#     OR realized_vol_10d > realized_vol_63d * 1.25
#
# Precedence (v2 §1C line 311):
#     crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
#
# `vol_crush` is wired via engine-pinned implementation decision using FRED VIXCLS-derived
# implied_vol_30d plus event_window_just_passed.
# ---------------------------------------------------------------------------


def evaluate_rising_vol(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C lines 250-254 `rising_vol` predicate at a single session.

    Returns False when ANY of the three inputs is NaN — strict cold-start
    contract (no silent "partial-input → True" substitution). Both limbs
    use strict ``>`` per spec text:

    * line 251 — ``atr_ratio > atr_ratio_threshold`` (1.15)
    * line 252 — ``realized_vol_short > realized_vol_long * realized_vol_ratio_threshold`` (1.25)
    * Combined: ATR limb OR realised-vol limb.

    The all-inputs-must-be-present contract is recorded in the
    engine-pinned implementation decision — spec §1C is silent on
    partial-NaN behavior so the conservative choice is "any NaN
    falsifies the rule" (matches recovery cold-start).
    """
    if dt not in features.atr_ratio.index:
        return False
    atr = features.atr_ratio.loc[dt]
    rv_short = features.realized_vol_short.loc[dt]
    rv_long = features.realized_vol_long.loc[dt]

    # Strict cold-start: any missing input falsifies the rule
    # (engine-pinned implementation decision — partial-NaN handling).
    if any(pd.isna(x) for x in (atr, rv_short, rv_long)):
        return False

    atr_limb = bool(atr > rules_config.atr_ratio_threshold)  # line 251
    rv_limb = bool(
        rv_short > rv_long * rules_config.realized_vol_ratio_threshold
    )  # line 252
    return atr_limb or rv_limb


def evaluate_vol_crush(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C `vol_crush` predicate at a single session (engine-pinned implementation decision).

    Rule (spec §1C):
      realized_vol_short < realized_vol_21d * vol_crush_realized_vol_ratio_threshold
      AND implied_vol_5d_change <= vol_crush_implied_vol_change_threshold
      AND event_window_just_passed

    Returns False when:
      - the Optional IV features are absent (no `implied_vol_30d` was
        supplied — `implied_vol_5d_change` is None),
      - the Optional `event_window_just_passed` series is absent (no
        event calendar was supplied),
      - any required input is NaN at ``dt`` (V1 §2.7 cold-start), or
      - ``dt`` is outside any of the input series' indices.

    All three guards collapse to the same outcome: when `vol_crush`'s
    extra data inputs are not wired, the rule simply does not fire and
    the precedence walker keeps the v1/`rising_vol` label.
    """
    iv_change = features.implied_vol_5d_change
    event_window = features.event_window_just_passed
    if iv_change is None or event_window is None:
        return False
    if (
        dt not in features.realized_vol_short.index
        or dt not in features.realized_vol_21d.index
        or dt not in iv_change.index
        or dt not in event_window.index
    ):
        return False

    rv_short = features.realized_vol_short.loc[dt]
    rv_mid = features.realized_vol_21d.loc[dt]
    iv_change_t = iv_change.loc[dt]
    if any(pd.isna(x) for x in (rv_short, rv_mid, iv_change_t)):
        return False

    rv_collapsed = bool(
        rv_short < rv_mid * rules_config.vol_crush_realized_vol_ratio_threshold
    )
    iv_falling_sharply = bool(
        iv_change_t <= rules_config.vol_crush_implied_vol_change_threshold
    )
    event_just_passed = bool(event_window.loc[dt])
    return rv_collapsed and iv_falling_sharply and event_just_passed


# v2 §1C line 311 ranking (lower index = higher precedence).
# `vol_crush` (index 1) was reserved-but-inert before engine-pinned implementation decision
# closure; it is now wired to a real predicate.
_V2_VOLATILITY_PRECEDENCE: tuple[str, ...] = (
    "crisis_vol",
    "vol_crush",
    "high_vol",
    "rising_vol",
    "low_vol",
    "normal_vol",
    "unknown",
)


def evaluate_v2_volatility_label(
    *,
    v1_label: str,
    features: VolatilityV2Features,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> str | None:
    """Apply v2 §1C volatility precedence on top of a v1 raw label.

    Returns the winning v2 label per the §1C line 311 ordering, or
    ``None`` when no v2 rule fires and the caller should keep ``v1_label``.

    Precedence (line 191): ``crisis_vol > vol_crush > high_vol >
    rising_vol > low_vol > normal_vol > unknown``.

    Dispatch order: `vol_crush` first (rank 1 — outranks high_vol /
    rising_vol; only `crisis_vol` outranks it). When `vol_crush` fires
    AND the v1 label is not `crisis_vol`, returns ``"vol_crush"``. Then
    `rising_vol` (rank 3): fires only when the v1 label is ranked
    strictly LOWER — i.e. v1 emitted ``low_vol`` / ``normal_vol`` /
    ``unknown``.
    """
    try:
        v1_rank = _V2_VOLATILITY_PRECEDENCE.index(v1_label)
    except ValueError:
        # Unknown v1 label — treat as lowest precedence.
        v1_rank = len(_V2_VOLATILITY_PRECEDENCE)

    # vol_crush (rank 1) — only crisis_vol outranks it. Fires when the
    # predicate is true AND v1 did not emit crisis_vol.
    vol_crush_rank = _V2_VOLATILITY_PRECEDENCE.index("vol_crush")
    if v1_rank >= vol_crush_rank and evaluate_vol_crush(
        features, dt=dt, rules_config=rules_config
    ):
        return "vol_crush"

    # rising_vol (rank 3) — fires only when v1 is ranked strictly lower.
    rising_vol_rank = _V2_VOLATILITY_PRECEDENCE.index("rising_vol")
    if v1_rank < rising_vol_rank:
        # v1 label outranks rising_vol (crisis_vol / vol_crush / high_vol).
        return None
    if evaluate_rising_vol(features, dt=dt, rules_config=rules_config):
        return "rising_vol"
    return None
