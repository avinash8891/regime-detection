from __future__ import annotations

import calendar
import datetime as dt

import pandas_market_calendars as mcal


_NYSE = mcal.get_calendar("NYSE")


def compute_monthly_options_expiry_anchor(*, year: int, month: int) -> dt.date:
    third_friday = _third_friday(year=year, month=month)
    trading_days = _nyse_trading_days(
        start_date=dt.date(year, month, 1),
        end_date=dt.date(year, month, calendar.monthrange(year, month)[1]),
    )
    if third_friday in trading_days:
        return third_friday

    prior_trading_days = [value for value in trading_days if value < third_friday]
    if not prior_trading_days:
        raise RuntimeError(f"No prior NYSE trading day found before third Friday {third_friday.isoformat()}")
    return prior_trading_days[-1]


def expand_trading_day_window(
    *,
    anchor_date: dt.date,
    lookback_trading_days: int,
    lookahead_trading_days: int,
) -> list[dt.date]:
    start_date = anchor_date - dt.timedelta(days=14)
    end_date = anchor_date + dt.timedelta(days=14)
    trading_days = _nyse_trading_days(start_date=start_date, end_date=end_date)
    try:
        anchor_idx = trading_days.index(anchor_date)
    except ValueError as exc:
        raise RuntimeError(f"Anchor date {anchor_date.isoformat()} is not an NYSE trading day") from exc

    window_start = anchor_idx - lookback_trading_days
    window_end = anchor_idx + lookahead_trading_days
    if window_start < 0 or window_end >= len(trading_days):
        raise RuntimeError(
            f"Trading-day window [{lookback_trading_days}, {lookahead_trading_days}] around {anchor_date.isoformat()} exceeded available NYSE session slice"
        )
    return trading_days[window_start : window_end + 1]


def build_monthly_options_expiry_anchors(*, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    anchors: list[dt.date] = []
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        anchors.append(compute_monthly_options_expiry_anchor(year=year, month=month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return anchors


def _third_friday(*, year: int, month: int) -> dt.date:
    month_weeks = calendar.monthcalendar(year, month)
    friday_column = calendar.FRIDAY
    friday_days = [week[friday_column] for week in month_weeks if week[friday_column] != 0]
    return dt.date(year, month, friday_days[2])


def _nyse_trading_days(*, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    schedule = _NYSE.schedule(start_date.isoformat(), end_date.isoformat())
    return [index.date() for index in schedule.index]
