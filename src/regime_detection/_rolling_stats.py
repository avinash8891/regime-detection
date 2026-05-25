"""Shared rolling-statistics helpers for cross-axis formulas."""

from __future__ import annotations

import pandas as pd

# Pandas/numpy sample-std convention (ddof=1).
_ZSCORE_DDOF = 1


def simple_moving_average(
    series: pd.Series,
    *,
    window: int,
    output_name: str | None = None,
) -> pd.Series:
    """Strict simple moving average: NaN until ``window`` observations exist."""
    if window <= 0:
        raise ValueError(f"window must be > 0; got {window}")
    out = series.astype(float).rolling(window=window, min_periods=window).mean()
    if output_name is not None:
        out = out.rename(output_name)
    return out


def period_return(
    series: pd.Series,
    *,
    periods: int,
    output_name: str | None = None,
) -> pd.Series:
    """Period return: ``series[t] / series[t-periods] - 1``."""
    if periods <= 0:
        raise ValueError(f"periods must be > 0; got {periods}")
    series = series.astype(float)
    out = series / series.shift(periods) - 1.0
    if output_name is not None:
        out = out.rename(output_name)
    return out


def rolling_change_zscore(
    series: pd.Series,
    *,
    change_window: int,
    normalizer_window: int,
    output_name: str | None = None,
) -> pd.Series:
    """Z-score of ``change_N`` against its own rolling mean/std."""
    if change_window <= 0:
        raise ValueError(f"change_window must be > 0; got {change_window}")
    if normalizer_window <= 0:
        raise ValueError(f"normalizer_window must be > 0; got {normalizer_window}")
    series = series.astype(float)
    change = series - series.shift(change_window)
    rolling = change.rolling(window=normalizer_window, min_periods=normalizer_window)
    mean = rolling.mean()
    std = rolling.std(ddof=_ZSCORE_DDOF)
    zscore = (change - mean) / std.where(std > 0)
    if output_name is not None:
        zscore = zscore.rename(output_name)
    return zscore
