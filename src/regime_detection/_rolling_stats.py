"""Shared rolling-statistics helpers for cross-axis formulas."""

from __future__ import annotations

import numpy as np
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


def rolling_ols_slope(series: pd.Series, *, window: int) -> pd.Series:
    """Rolling OLS slope of ``series`` vs a unit trading-day index.

    Closed-form OLS slope via centered moments: ``cov(x, y) / var(x)`` where
    ``x = [0, 1, ..., window-1]``. Returns NaN until ``window`` non-NaN
    observations are available; any NaN in the window propagates to NaN.

    The centered form is the canonical implementation: it subtracts means
    BEFORE multiplying, dramatically reducing intermediate dynamic range
    compared with the algebraically equivalent normal-equations form
    ``(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)``. The uncentered form can suffer
    catastrophic cancellation for near-zero slopes feeding sign-sensitive
    predicates (e.g. network_fragility's ``avg_pairwise_corr_slope_21d > 0``),
    which is part of the V1 wire and must remain bit-stable.

    Single home for OLS-slope math — formerly duplicated in
    ``credit_funding._rolling_ols_slope`` and
    ``network_fragility_rules._rolling_ols_slope_series``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    series = series.astype(float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = float((x_centered**2).sum())

    def _slope(window_arr: np.ndarray) -> float:
        if np.isnan(window_arr).any():
            return float("nan")
        y_mean = window_arr.mean()
        return float((x_centered * (window_arr - y_mean)).sum() / x_var)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)
