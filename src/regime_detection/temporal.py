from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd


def parse_date_series(
    values: Iterable[object],
    *,
    field_name: str,
    context: str,
    nullable: bool = False,
) -> pd.Series:
    """Parse a canonical date-only column into ``datetime.date`` values."""

    raw = pd.Series(values)
    missing = raw.isna()
    parsed = pd.to_datetime(raw, errors="coerce")
    bad = parsed.isna() & (~missing if nullable else pd.Series([True] * len(raw)))
    if bad.any():
        bad_values = sorted({str(value) for value in raw.loc[bad].tolist()})
        raise ValueError(
            f"{context} contains malformed {field_name} values: {bad_values}"
        )
    if missing.any() and not nullable:
        raise ValueError(f"{context} contains missing {field_name} values")

    out = parsed.dt.date.astype("object")
    if nullable:
        out = out.where(~missing, None)
    return out


def parse_datetime_series(
    values: Iterable[object],
    *,
    field_name: str,
    context: str,
) -> pd.Series:
    """Parse a timestamp/date column with consistent error messages."""

    raw = pd.Series(values)
    missing = raw.isna()
    parsed = pd.to_datetime(raw, errors="coerce")
    bad = parsed.isna() & ~missing
    if bad.any():
        bad_values = sorted({str(value) for value in raw.loc[bad].tolist()})
        raise ValueError(
            f"{context} contains malformed {field_name} values: {bad_values}"
        )
    if missing.any():
        raise ValueError(f"{context} contains missing {field_name} values")
    return parsed


def parse_datetime_index(
    values: Iterable[object],
    *,
    field_name: str,
    context: str,
) -> pd.DatetimeIndex:
    """Parse a date-like column into the engine's UTC-naive session index."""

    return pd.DatetimeIndex(
        parse_datetime_series(values, field_name=field_name, context=context)
    )
