from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal
from zoneinfo import ZoneInfo


# ±-window for nyse_neighbors: 10 calendar days comfortably covers the
# longest NYSE closure stretch (weekend + multi-day holiday, e.g. Christmas
# + NYD ≈ 5 days). Tighter values risk failing on edge cases.
NYSE_NEIGHBOR_WINDOW_DAYS = 10


@dataclass(frozen=True)
class TradingDayNeighbors:
    prev_trading_day: date
    next_trading_day: date


def nyse_calendar() -> mcal.MarketCalendar:
    return mcal.get_calendar("NYSE")


@lru_cache(maxsize=None)
def nyse_sessions_between(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(nyse_calendar().schedule(start_date=start_date, end_date=end_date).index.date)


def _as_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("America/New_York").date()
        if value == value.normalize():
            return value.date()
        # tz-naive pandas Timestamps are ambiguous; require tz-aware or plain date.
        raise TypeError("tz-naive pandas Timestamp is ambiguous; pass a date or a tz-aware Timestamp (America/New_York recommended)")
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            # Interpret date-like inputs in US/Eastern for NYSE calendar alignment.
            return value.astimezone(ZoneInfo("America/New_York")).date()
        # tz-naive datetimes are ambiguous; require callers to provide either a plain date
        # or a timezone-aware datetime.
        raise TypeError("tz-naive datetime is ambiguous; pass a date or a tz-aware datetime (America/New_York recommended)")
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    raise TypeError(f"Expected date-like value, got {type(value).__name__}")


def as_date(value: object) -> date:
    """
    Public coercion helper: accept common date-like inputs (date, datetime, pandas Timestamp)
    and normalize to a plain `datetime.date`.
    """
    return _as_date(value)


def is_nyse_trading_day(d: date) -> bool:
    d = _as_date(d)
    return len(nyse_sessions_between(d, d)) > 0


def nyse_neighbors(d: date) -> TradingDayNeighbors:
    d = _as_date(d)

    # Get a small window around the target date and pick nearest trading days.
    as_ts = pd.Timestamp(d)
    start = (as_ts - pd.Timedelta(days=NYSE_NEIGHBOR_WINDOW_DAYS)).date()
    end = (as_ts + pd.Timedelta(days=NYSE_NEIGHBOR_WINDOW_DAYS)).date()
    sessions = pd.DatetimeIndex(nyse_sessions_between(start, end))
    if sessions.empty:
        raise RuntimeError("NYSE calendar returned empty schedule window")

    prev_sessions = sessions[sessions < as_ts]
    next_sessions = sessions[sessions > as_ts]
    if prev_sessions.empty or next_sessions.empty:
        raise RuntimeError("Unable to find NYSE neighbor sessions around date")

    return TradingDayNeighbors(
        prev_trading_day=prev_sessions.max().date(),
        next_trading_day=next_sessions.min().date(),
    )


def require_nyse_trading_day(as_of_date: date) -> None:
    as_of_date = _as_date(as_of_date)
    if is_nyse_trading_day(as_of_date):
        return
    neighbors = nyse_neighbors(as_of_date)
    raise ValueError(
        "as_of_date must be an NYSE trading day. "
        f"Got {as_of_date.isoformat()}. "
        f"Nearest prior trading day: {neighbors.prev_trading_day.isoformat()}. "
        f"Nearest next trading day: {neighbors.next_trading_day.isoformat()}."
    )
