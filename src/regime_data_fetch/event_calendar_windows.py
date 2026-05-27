from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportMissingTypeStubs=false

import datetime as dt

import pandas_market_calendars as mcal

from regime_data_fetch.event_calendar_models import EventCalendarFetchError

_NYSE = mcal.get_calendar("NYSE")


def us_general_election_date(year: int) -> dt.date:
    first_november = dt.date(year, 11, 1)
    days_until_monday = (0 - first_november.weekday()) % 7
    first_monday = first_november + dt.timedelta(days=days_until_monday)
    return first_monday + dt.timedelta(days=1)


def expand_nyse_window_for_scheduled_event(
    *,
    anchor_date: dt.date,
    lookback_trading_days: int,
    lookahead_trading_days: int,
) -> list[dt.date]:
    start_date = anchor_date - dt.timedelta(days=14)
    end_date = anchor_date + dt.timedelta(days=14)
    schedule = _NYSE.schedule(start_date.isoformat(), end_date.isoformat())
    trading_days = [index.date() for index in schedule.index]
    try:
        anchor_idx = trading_days.index(anchor_date)
    except ValueError as exc:
        raise EventCalendarFetchError(
            f"Scheduled event date {anchor_date.isoformat()} is not an NYSE trading day"
        ) from exc

    window_start = anchor_idx - lookback_trading_days
    window_end = anchor_idx + lookahead_trading_days
    if window_start < 0 or window_end >= len(trading_days):
        raise EventCalendarFetchError(
            f"NYSE window [{lookback_trading_days}, {lookahead_trading_days}] around {anchor_date.isoformat()} exceeded available trading-day slice"
        )
    return trading_days[window_start : window_end + 1]
