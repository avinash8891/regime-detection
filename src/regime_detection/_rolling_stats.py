"""Shared rolling-statistics helpers for cross-axis formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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

    The centered form subtracts means BEFORE multiplying, reducing intermediate
    dynamic range compared with the uncentered normal-equations form
    ``(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`` which can suffer catastrophic
    cancellation for near-zero slopes feeding sign-sensitive predicates
    (e.g. network_fragility's ``avg_pairwise_corr_slope_21d > 0`` V2 rule).
    Note: the V1 wire hardcodes ``network_fragility`` to a placeholder — this
    predicate is in the V2 classification path only.

    Fully vectorized via ``sliding_window_view`` — no per-window Python loop.

    Single home for OLS-slope math — formerly duplicated in
    ``credit_funding._rolling_ols_slope`` and
    ``network_fragility_rules._rolling_ols_slope_series``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=series.index)

    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_var = float((x_centered**2).sum())

    windows = sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)
    if valid.any():
        vw = windows[valid]
        y_centered = vw - vw.mean(axis=1, keepdims=True)
        slopes = (y_centered @ x_centered) / x_var
        out[window - 1 + np.flatnonzero(valid)] = slopes
    return pd.Series(out, index=series.index)


def rolling_stability(series: pd.Series, *, window: int) -> pd.Series:
    """Rolling coefficient-of-variation (std/mean, ddof=0) over a trailing window.

    Returns NaN until ``window`` non-NaN observations are available; any NaN
    in the window propagates to NaN. Rows where the rolling mean is zero
    produce NaN (undefined CV). Used by network_fragility §3.5
    ``diversified_normal`` to assess effective-rank stability.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(out, index=series.index)

    windows = sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1)
    if valid.any():
        vw = windows[valid]
        means = vw.mean(axis=1)
        stabilities = np.full(len(vw), np.nan, dtype=float)
        nonzero = means != 0.0
        if nonzero.any():
            stabilities[nonzero] = vw[nonzero].std(axis=1, ddof=0) / means[nonzero]
        out[window - 1 + np.flatnonzero(valid)] = stabilities
    return pd.Series(out, index=series.index)


def rolling_drawdown(close: pd.Series, *, window: int) -> pd.Series:
    """Rolling drawdown of ``close`` from its trailing ``window``-day peak.

    Returns ``close / peak - 1`` (negative; 0 at fresh highs). NaN where the
    peak window hasn't filled or where the peak is non-positive (avoids
    division by zero on adjusted-price series with non-positive values).
    """
    peak = close.rolling(window=window, min_periods=window).max()
    drawdown = close / peak - 1.0
    return drawdown.where(peak > 0).astype(float)
