from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

_PMI_PATTERN = re.compile(
    r"(?P<label>[A-Za-z ]+PMI)\s+at\s+(?P<value>\d+(?:\.\d+)?)%", re.IGNORECASE
)


def extract_ism_pmi_value(html: str, *, label: str) -> float:
    for match in _PMI_PATTERN.finditer(html):
        if match.group("label").strip().lower() == label.strip().lower():
            return float(match.group("value"))
    raise ValueError(f"Could not find {label!r} in ISM page")


def release_timestamp_for(
    *, year: int, month: int, business_day_index: int
) -> dt.datetime:
    if business_day_index < 1:
        raise ValueError("business_day_index must be >= 1")

    ts = _nth_business_day(year=year, month=month, n=business_day_index)
    return dt.datetime(
        ts.year, ts.month, ts.day, 10, 0, tzinfo=ZoneInfo("America/New_York")
    )


def _nth_business_day(*, year: int, month: int, n: int) -> dt.date:
    current = dt.date(year, month, 1)
    seen = 0
    while current.month == month:
        if current.weekday() < 5:
            seen += 1
            if seen == n:
                return current
        current += dt.timedelta(days=1)
    raise ValueError(f"Month {year:04d}-{month:02d} has fewer than {n} business days")
