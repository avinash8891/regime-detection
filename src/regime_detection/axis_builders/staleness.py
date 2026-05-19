from __future__ import annotations

import pandas as pd

_STALENESS_SENTINEL = 10**9


def _calendar_staleness_days_series(
    series: pd.Series | None, session_index: pd.Index
) -> pd.Series:
    if series is None:
        return pd.Series(_STALENESS_SENTINEL, index=session_index, dtype="int64")
    valid_dates = pd.DatetimeIndex(series.dropna().index).sort_values()
    if valid_dates.empty:
        return pd.Series(_STALENESS_SENTINEL, index=session_index, dtype="int64")

    sessions = pd.DatetimeIndex(session_index)
    source_positions = valid_dates.searchsorted(sessions, side="right") - 1
    last_valid_date = pd.Series(pd.NaT, index=session_index, dtype="datetime64[ns]")
    has_observation = source_positions >= 0
    last_valid_date.loc[has_observation] = valid_dates[source_positions[has_observation]]
    delta_days = (pd.Series(sessions, index=session_index) - last_valid_date).dt.days
    return delta_days.fillna(_STALENESS_SENTINEL).astype("int64")


def _trading_staleness_series(
    series: pd.Series | None, session_index: pd.Index
) -> pd.Series:
    if series is None:
        return pd.Series(_STALENESS_SENTINEL, index=session_index, dtype="int64")
    aligned = series.reindex(session_index)
    session_pos = pd.Series(
        range(len(session_index)), index=session_index, dtype="int64"
    )
    valid = aligned.notna()
    last_valid_pos = (
        session_pos.where(valid).ffill().fillna(-_STALENESS_SENTINEL).astype("int64")
    )
    return (session_pos - last_valid_pos).astype("int64")


def _safe_float(series: pd.Series, dt: pd.Timestamp) -> float:
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)
