from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import pandas_market_calendars as mcal


@dataclass(frozen=True)
class TradingDayNeighbors:
    prev_trading_day: date
    next_trading_day: date


def nyse_calendar() -> mcal.MarketCalendar:
    return mcal.get_calendar("NYSE")


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    raise TypeError(f"Expected date-like value, got {type(value).__name__}")


def as_date(value: object) -> date:
    """
    Public coercion helper: accept common date-like inputs (date, datetime, pandas Timestamp)
    and normalize to a plain `datetime.date`.
    """
    return _as_date(value)


def is_nyse_trading_day(d: date) -> bool:
    d = _as_date(d)
    cal = nyse_calendar()
    schedule = cal.schedule(start_date=d, end_date=d)
    return not schedule.empty


def nyse_neighbors(d: date) -> TradingDayNeighbors:
    d = _as_date(d)
    cal = nyse_calendar()

    # Get a small window around the target date and pick nearest trading days.
    as_ts = pd.Timestamp(d)
    start = (as_ts - pd.Timedelta(days=10)).date()
    end = (as_ts + pd.Timedelta(days=10)).date()
    schedule = cal.schedule(start_date=start, end_date=end)
    if schedule.empty:
        raise RuntimeError("NYSE calendar returned empty schedule window")

    sessions = schedule.index

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
