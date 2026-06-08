"""Shared per-day scalar plumbing for axis classifier rule layers.

Consolidates helpers that were previously duplicated 2–4× across the V2
axis rule files. All consumers were behaviourally equivalent for
float-dtype series; the canonical forms here are the most-defensive of the
existing duplicates (explicit ``pd.isna`` guards on scalar reads) so that
the consolidation is a strict superset, never less safe.

Companion module to ``_rolling_stats.py`` (rolling-statistics helpers).
Both modules are package-private (leading underscore on the filename) but
export public functions used by every V2 axis classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def any_nan(*values: float) -> bool:
    """True if any of ``values`` is NaN.

    Used by rule predicates as a single cold-start gate before applying
    threshold comparisons.
    """
    return any(np.isnan(v) for v in values)


def is_nan(value: float) -> bool:
    """True if ``value`` is NaN.

    Single-argument variant for predicates that gate on one input at a
    time (e.g. ``volume_liquidity_rules._gap_history_unavailable``).
    """
    return bool(np.isnan(value))


def scalar_at(series: pd.Series, dt: pd.Timestamp) -> float:
    """Read ``series[dt]`` as a float; return NaN if ``dt`` missing or value NaN.

    Canonical form is the explicit ``pd.isna`` guard from credit_funding /
    inflation_growth / monetary_pressure. network_fragility_rules's pre-
    consolidation version omitted the guard but is behaviourally identical
    for float-dtype series (``float(np.nan) == nan``); the guard is
    defensive against accidental object-dtype callers.
    """
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)


def scalar_at_lag(series: pd.Series, dt: pd.Timestamp, lag: int) -> float:
    """Read ``series`` positionally ``lag`` rows before ``dt``.

    Returns NaN if ``dt`` is not in the index, if the lagged position falls
    before the start, or if the value at the lagged position is NaN.

    Positional (``.iloc``) lookup rather than label-based — the lag is
    counted in row offsets in the series' own index, not calendar days. The
    caller is responsible for ensuring the series' index has the expected
    cadence (typically NYSE trading days).
    """
    if dt not in series.index:
        return float("nan")
    pos = series.index.get_loc(dt)
    if pos - lag < 0:
        return float("nan")
    val = series.iloc[pos - lag]
    if pd.isna(val):
        return float("nan")
    return float(val)


def ev_float(x: float) -> float:
    """Round a float to 8 decimals for evidence-dict serialisation.

    All V1 axes write rule-evidence values as floats rounded to 8 decimals
    so the V1 frozen-replay byte-identity contract holds across runs.
    Centralising the rounding here removes drift risk if any caller
    accidentally rounded to a different precision.
    """
    return round(float(x), 8)
