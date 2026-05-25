from __future__ import annotations

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
    """Parse a date-like column into a normalized NYSE session index.

    Naive date-like values are interpreted as already being NYSE session dates.
    Timezone-aware values are converted to America/New_York before dropping
    timezone metadata, so the returned index stays comparable with the engine's
    existing tz-naive session indexes while remaining accepted by
    ``calendar.as_date`` as a session-date timestamp.
    """

    raw = pd.Series(values)
    missing = raw.isna()
    parsed_values: list[pd.Timestamp | pd.NaT] = []
    for value in raw:
        if pd.isna(value):
            parsed_values.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            parsed_values.append(pd.NaT)
            continue
        if pd.isna(timestamp):
            parsed_values.append(pd.NaT)
        elif timestamp.tzinfo is None:
            parsed_values.append(timestamp.normalize())
        else:
            parsed_values.append(
                timestamp.tz_convert("America/New_York").tz_localize(None).normalize()
            )

    parsed = pd.Series(parsed_values)
    bad = parsed.isna() & ~missing
    if bad.any():
        bad_values = sorted({str(value) for value in raw.loc[bad].tolist()})
        raise ValueError(
            f"{context} contains malformed {field_name} values: {bad_values}"
        )
    if missing.any():
        raise ValueError(f"{context} contains missing {field_name} values")
    return pd.DatetimeIndex(parsed_values)
