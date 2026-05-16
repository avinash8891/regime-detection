"""Shared rolling-statistics helpers used by multiple V2 axes.

Lives outside any individual axis module so the helper is neutral —
neither §2A (`monetary_pressure`) nor §2C (`credit_funding`) "owns" the
formula. AGENTS rule B: one home per concept.

Each function takes the window lengths as explicit parameters so spec
citations live at the call site (e.g. §2A pins 63d-change / 1260d-
normalizer; §2C pins 21d-change / 1260d-normalizer).
"""
from __future__ import annotations

import pandas as pd

# Pandas/numpy sample-std convention. Pinned identically in
# `regime_detection.monetary_pressure._ZSCORE_DDOF` and consumed here so
# both §2A and §2C consumers normalize with the same ddof.
_ZSCORE_DDOF = 1


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` periods."""
    return series.rolling(window).mean()


def period_return(series: pd.Series, periods: int) -> pd.Series:
    """Percentage return over ``periods`` look-back: series / series.shift(periods) - 1."""
    return series / series.shift(periods) - 1


def rolling_change_zscore(
    series: pd.Series,
    *,
    change_window: int,
    normalizer_window: int,
    output_name: str | None = None,
) -> pd.Series:
    """Z-score of ``change_N`` against the rolling mean/std of ``change_N``.

    ``change_N[t] = series[t] - series[t-change_window]``, then z-score
    normalises by the rolling mean / std over ``normalizer_window`` of
    the change series itself (not the level series — that's the
    spec convention for §2A line 896 + §2C line 2054).

    Constant-change windows produce ``std == 0`` which is masked to NaN
    via ``.where(std > 0)``.

    Parameters
    ----------
    series
        Input level series (e.g. yield or index level).
    change_window
        Lookback for ``change[t] = series[t] - series[t-N]``. Must be > 0.
    normalizer_window
        Rolling window for the mean / std of the change series. Must be > 0.
    output_name
        Optional ``.name`` for the returned Series. Passed through for
        downstream debugging / wire emission.
    """
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
