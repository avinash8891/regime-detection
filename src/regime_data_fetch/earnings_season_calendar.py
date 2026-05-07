from __future__ import annotations

import calendar
import datetime as dt


QUARTER_START_MONTHS = (1, 4, 7, 10)
WINDOW_LENGTH_CALENDAR_DAYS = 35


def compute_earnings_season_window(*, year: int, quarter_start_month: int) -> tuple[dt.date, dt.date]:
    if quarter_start_month not in QUARTER_START_MONTHS:
        raise ValueError(f"quarter_start_month must be one of {QUARTER_START_MONTHS}")

    start_date = _second_monday(year=year, month=quarter_start_month)
    end_date = start_date + dt.timedelta(days=WINDOW_LENGTH_CALENDAR_DAYS)
    return start_date, end_date


def build_earnings_season_windows(*, start_date: dt.date, end_date: dt.date) -> list[tuple[dt.date, dt.date]]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    windows: list[tuple[dt.date, dt.date]] = []
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        if month in QUARTER_START_MONTHS:
            windows.append(compute_earnings_season_window(year=year, quarter_start_month=month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return windows


def is_in_earnings_season(*, as_of_date: dt.date) -> bool:
    start_date, end_date = compute_earnings_season_window(
        year=as_of_date.year,
        quarter_start_month=_candidate_quarter_start_month(as_of_date.month),
    )
    if start_date <= as_of_date <= end_date:
        return True

    previous_year, previous_month = _previous_quarter_start(start_date.year, start_date.month)
    previous_start, previous_end = compute_earnings_season_window(
        year=previous_year,
        quarter_start_month=previous_month,
    )
    return previous_start <= as_of_date <= previous_end


def _second_monday(*, year: int, month: int) -> dt.date:
    weeks = calendar.monthcalendar(year, month)
    monday_column = calendar.MONDAY
    mondays = [week[monday_column] for week in weeks if week[monday_column] != 0]
    return dt.date(year, month, mondays[1])


def _candidate_quarter_start_month(month: int) -> int:
    if month >= 10:
        return 10
    if month >= 7:
        return 7
    if month >= 4:
        return 4
    return 1


def _previous_quarter_start(year: int, month: int) -> tuple[int, int]:
    idx = QUARTER_START_MONTHS.index(month)
    if idx == 0:
        return year - 1, QUARTER_START_MONTHS[-1]
    return year, QUARTER_START_MONTHS[idx - 1]
