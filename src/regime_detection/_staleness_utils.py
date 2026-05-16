"""Staleness computation helpers for axis-series classifiers.

Split out from axis_series.py so that individual axis modules can use these
utilities without creating circular imports back into ``axis_series.py``.
"""
from __future__ import annotations

import pandas as pd

_STALENESS_SENTINEL = 10**9


def calendar_staleness_days_series(
    series: pd.Series | None, session_index: pd.Index
) -> pd.Series:
    """Return a series of calendar-day staleness counts aligned to ``session_index``.

    For each session, the value is the number of calendar days since the last
    non-NaN value in ``series``. Returns ``_STALENESS_SENTINEL`` for all
    sessions when ``series`` is None or has no prior valid value.
    """
    if series is None:
        return pd.Series(_STALENESS_SENTINEL, index=session_index, dtype="int64")
    aligned = series.reindex(session_index)
    last_valid_date = pd.Series(pd.NaT, index=session_index, dtype="datetime64[ns]")
    valid = aligned.notna()
    last_valid_date.loc[valid] = session_index[valid]
    last_valid_date = last_valid_date.ffill()
    delta_days = (pd.Series(session_index, index=session_index) - last_valid_date).dt.days
    return delta_days.fillna(_STALENESS_SENTINEL).astype("int64")


def trading_staleness_series(
    series: pd.Series | None, session_index: pd.Index
) -> pd.Series:
    """Return a series of trading-session staleness counts aligned to ``session_index``.

    For each session, the value is the number of trading sessions since the last
    non-NaN value in ``series``. Returns ``_STALENESS_SENTINEL`` for all
    sessions when ``series`` is None or has no prior valid value.
    """
    if series is None:
        return pd.Series(_STALENESS_SENTINEL, index=session_index, dtype="int64")
    aligned = series.reindex(session_index)
    session_pos = pd.Series(range(len(session_index)), index=session_index, dtype="int64")
    valid = aligned.notna()
    last_valid_pos = session_pos.where(valid).ffill().fillna(-_STALENESS_SENTINEL).astype("int64")
    return (session_pos - last_valid_pos).astype("int64")


def safe_float(series: pd.Series, dt: pd.Timestamp) -> float:
    """Return ``float(series.loc[dt])``, or ``nan`` if missing or NaN."""
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)
